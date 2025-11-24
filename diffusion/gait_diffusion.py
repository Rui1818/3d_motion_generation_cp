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

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
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
        super(GaitDiffusionModel, self).__init__(
            **kwargs,
        )

    def masked_l2(self, a, b):
        #TODO: adjust loss to keypoint type
        bs, n, c = a.shape
        """
        #euclidean distance
        if c==132:
            loss=(a - b)**2
            loss=sum_flat(loss)/(n*c)
            
            loss = torch.mean(
                torch.norm(
                    (a - b).reshape(-1, 6),
                    2,
                    1,
                )
            )
        else:
            
            loss = torch.mean(
                torch.norm(
                    (a - b).reshape(-1, 3),
                    2,
                    1,
                )
            )
        """
        loss=(a - b)**2
        loss=sum_flat(loss)/(n*c)

        return loss

    def masked_euclidean_distance(self, a, b):
        """
        Calculates the mean Euclidean distance loss between predicted and target vectors.
        This is often a better choice for structured data like joint positions.
        """
        bs, n, c = a.shape

        # Determine the dimension of the individual items (e.g., 3 for xyz, 6 for 6D rot)
        # This logic assumes features are groups of 3 or 6.
        if c == 132:
            item_dim = 6
        elif c == 72 or c == 3:
            item_dim = 3
        else:
            raise ValueError(f"Unexpected features_dim: {c}")

        # Reshape to group features into vectors: (batch, num_vectors, item_dim)
        a_vecs = a.reshape(bs, -1, item_dim)
        b_vecs = b.reshape(bs, -1, item_dim)

        # Calculate the L2 norm (Euclidean distance) for each vector
        euclidean_distances = torch.norm(a_vecs - b_vecs, dim=-1)

        # Return the mean distance per batch item.
        return euclidean_distances.mean(dim=-1)

    def training_losses(
        self, model, x_start, t, cond, model_kwargs=None, noise=None, dataset=None
    ):

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

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
            model_output = model(x_t, self._scale_timesteps(t), cond, **model_kwargs)

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

            terms["simple_mse"] = self.masked_l2(
                target,
                model_output,
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
                    terms['rot_vel_mse'] = self.masked_l2(
                        target_rot_vel,
                        model_output_rot_vel,
                    )
                    terms['transl_vel_mse'] = self.masked_l2(
                        target_transl_vel,
                        model_output_transl_vel,
                    )
                else:
                    #keypoints only
                    target_vel = target[:, 1:, :]- target[:, :-1, :]
                    model_output_vel = model_output[:, 1:, :]- model_output[:, :-1, :]
                    terms['rot_vel_mse'] = self.masked_l2(
                        target_vel,
                        model_output_vel,
                    )



            
            terms["loss"] = terms["simple_mse"] + terms.get("vb", 0.0) 
            + self.lambda_rot_vel * terms.get("rot_vel_mse", 0.0) 
            + self.lambda_transl_vel * terms.get("transl_vel_mse", 0.0)

        else:
            raise NotImplementedError(self.loss_type)

        return terms