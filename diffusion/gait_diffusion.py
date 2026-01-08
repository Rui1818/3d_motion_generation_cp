"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model

import torch
import torch as th
import fastdtw
import numpy as np
from utils.metrics import pose_distance_metric

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from utils.soft_dtw_cuda import SoftDTW
def sum_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Takes the sum over all non-batch dimensions.
    
    For a tensor of shape (B, F, D), it will sum over F and D,
    resulting in a tensor of shape (B,).

    :param tensor: A PyTorch tensor where the first dimension is the batch size.
    :return: A tensor of shape (batch_size,) with the summed values.
    """
    # The dimensions to sum over are all dimensions starting from the second one (index 1).
    return tensor.sum(dim=tuple(range(1, tensor.dim())))




class GaitDiffusionModel(GaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):
        self.lambda_rot_vel = kwargs.pop("lambda_rot_vel", 0.0)
        self.lambda_transl_vel = kwargs.pop("lambda_transl_vel", 0.0)
        self.soft_dtw_gamma = kwargs.pop("soft_dtw_gamma", 1.0)
        self.soft_dtw_normalize = kwargs.pop("soft_dtw_normalize", False)
        self.loss_func = kwargs.pop("loss_func", "mse")  # "mse" or "softdtw"
        super(GaitDiffusionModel, self).__init__(
            **kwargs,
        )
        # Initialize SoftDTW module (will be created on first use to ensure correct device)
        self._soft_dtw = None

    def masked_l2(self, a, b, seqlen):
        """
        Computes the masked L2 loss.
        The loss is computed only on the frames before the sequence length.
        """
        batch, frames, features = a.shape
        # Create a mask for the sequences
        indices = torch.arange(frames, device=a.device).unsqueeze(0)

        # mask is a boolean tensor of shape [B, N]
        mask = indices < seqlen.unsqueeze(1)

        # Expand mask to match the shape of a and b: [B, N, C]
        mask_expanded = mask.unsqueeze(-1).expand_as(a)

        # Compute squared error and apply the mask
        loss = (a - b) ** 2
        masked_loss = loss * mask_expanded

        # Normalize by the number of valid elements
        # seqlen has shape [B], so we need to unsqueeze it for broadcasting
        num_valid_elements = seqlen.unsqueeze(1) * features
        # Avoid division by zero for sequences of length 0
        num_valid_elements = torch.max(num_valid_elements, torch.ones_like(num_valid_elements))

        # Sum the loss over frames and features, then normalize
        return sum_flat(masked_loss) / num_valid_elements.squeeze(1)

    def masked_soft_dtw(self, a, b, seqlen):
        """
        Computes the masked Soft-DTW loss.
        The reference sequence (a) is masked to only use frames before the sequence length.

        Args:
            a: Reference sequence (batch, frames, features) - will be masked
            b: Generated sequence (batch, frames, features)
            seqlen: Sequence lengths for each batch element (batch,)

        Returns:
            Soft-DTW loss per batch element (batch,)
        """
        # Initialize SoftDTW module if not already done
        if self._soft_dtw is None:
            use_cuda = a.is_cuda
            self._soft_dtw = SoftDTW(
                use_cuda=False,
                gamma=self.soft_dtw_gamma,
                normalize=True,
                bandwidth=None
            )
            # Move to same device as input
            if hasattr(self._soft_dtw, 'to'):
                self._soft_dtw = self._soft_dtw.to(a.device)

        batch, frames, features = a.shape

        # Create masked versions of the reference sequence
        # We'll extract only the valid frames for each sequence in the batch
        losses = []

        for i in range(batch):
            # Get the valid length for this sequence
            valid_len = int(seqlen[i].item())

            # Handle edge case of zero-length sequences
            if valid_len <= 0:
                losses.append(torch.tensor(0.0, device=a.device, dtype=a.dtype))
                continue

            # Extract valid frames from reference (a) - masked to valid_len
            # Use full sequence from generated (b) - not masked
            a_valid = a[i:i+1, :valid_len, :]  # Shape: (1, valid_len, features)
            b_full = b[i:i+1, :, :]             # Shape: (1, frames, features)

            # Compute Soft-DTW between masked reference and full generated sequence
            loss = self._soft_dtw(a_valid, b_full)
            losses.append(loss.squeeze())

        # Stack losses into a tensor of shape (batch,)
        return torch.stack(losses)

    def training_losses(
        self, model, x_start, t, cond, model_kwargs=None, noise=None, dataset=None, eval_dtw=False
    ):

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        seq_len = model_kwargs.get("y", {}).get("seq_len", None)
        # Now you can use seq_len in your loss calculation

        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        #choose the loss function
        loss_func = self.masked_soft_dtw if self.loss_func == "softdtw" else self.masked_l2

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), cond, **{k: v for k, v in model_kwargs.items() if k != 'y'})

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape

            terms["simple_mse"] = loss_func(
                target,
                model_output,
                seq_len,
            )

            if self.lambda_rot_vel > 0.0 or self.lambda_transl_vel > 0.0:
                #velocity loss
                _,_,c=target.shape
                if c==135:
                    #rotation representation+translation
                    target_rot_vel = target[:, 1:, 3:]- target[:, :-1, 3:]
                    model_output_rot_vel = model_output[:, 1:, 3:]- model_output[:, :-1, 3:]
                    target_transl_vel = target[:, 1:, :3]- target[:, :-1, :3]
                    model_output_transl_vel = model_output[:, 1:, :3]- model_output[:, :-1, :3]
                    terms['rot_vel_mse'] = loss_func(
                        target_rot_vel,
                        model_output_rot_vel,
                        seq_len - 1, # Velocity has one less frame
                    )
                    terms['transl_vel_mse'] = loss_func(
                        target_transl_vel,
                        model_output_transl_vel,
                        seq_len - 1, # Velocity has one less frame
                    )
                else:
                    #keypoints only
                    target_vel = target[:, 1:, :]- target[:, :-1, :]
                    model_output_vel = model_output[:, 1:, :]- model_output[:, :-1, :]
                    terms['rot_vel_mse'] = loss_func(
                        target_vel,
                        model_output_vel,
                        seq_len - 1, # Velocity has one less frame
                    )



            
            terms["loss"] = (terms["simple_mse"] + terms.get("vb", 0.0) 
            + self.lambda_rot_vel * terms.get("rot_vel_mse", 0.0) 
            + self.lambda_transl_vel * terms.get("transl_vel_mse", 0.0))
            
            if eval_dtw:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart = model_output
                elif self.model_mean_type == ModelMeanType.EPSILON:
                    pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
                else:
                    pred_xstart = model_output

                dtw_losses = []
                dtw_losses_geodesic = []
                pred_np = pred_xstart.detach().cpu().numpy()
                target_np = x_start.detach().cpu().numpy()
                dist_fn = lambda x, y: np.linalg.norm(x - y)
                dist_geodesic = pose_distance_metric

                for i in range(len(pred_np)):
                    sl = int(seq_len[i].item()) if seq_len is not None else pred_np.shape[1]
                    d, path = fastdtw.fastdtw(pred_np[i, :], target_np[i, :sl], dist=dist_fn)
                    dtw_losses.append(d/len(path))  # normalize by path length

                    if target_np.shape[2]==135:
                        #calculate geodesic distance only on rotation part
                        target_np_rot = target_np[i, :sl, 3:]
                        pred_np_rot = pred_np[i, :sl, 3:]
                        d_geodesic, path = fastdtw.fastdtw(pred_np_rot, target_np_rot, dist=dist_geodesic)
                        dtw_losses_geodesic.append(d_geodesic/len(path))  # normalize by path length
                    
                
                terms["dtw_loss"] = torch.tensor(np.mean(dtw_losses), device=x_start.device)
                if target_np.shape[2]==135:
                    terms["dtw_loss_geodesic"] = torch.tensor(np.mean(dtw_losses_geodesic), device=x_start.device)

        else:
            raise NotImplementedError(self.loss_type)

        return terms