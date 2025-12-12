import torch
import time
import sys
import os

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.soft_dtw_cuda import SoftDTW

def masked_soft_dtw_impl(soft_dtw_module, a, b, seqlen, normalize=False):
    """
    Re-implementation of GaitDiffusionModel.masked_soft_dtw logic.
    This function iterates over the batch and computes SoftDTW for each pair
    using the valid length for 'a' and full length for 'b'.
    """
    batch, frames, features = a.shape
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
        # Note: In GaitDiffusionModel, 'a' is reference (masked), 'b' is generated (full)
        a_valid = a[i:i+1, :valid_len, :]  # Shape: (1, valid_len, features)
        b_full = b[i:i+1, :, :]             # Shape: (1, frames, features)

        # Compute Soft-DTW between masked reference and full generated sequence
        if normalize:
            # Manually compute normalized SoftDTW to avoid shape mismatch issues
            # in the SoftDTW module's internal batching (torch.cat) when lengths differ.
            loss_xy = soft_dtw_module(a_valid, b_full)
            loss_xx = soft_dtw_module(a_valid, a_valid)
            loss_yy = soft_dtw_module(b_full, b_full)
            loss = loss_xy - 0.5 * (loss_xx + loss_yy)
        else:
            loss = soft_dtw_module(a_valid, b_full)
        losses.append(loss.squeeze())

    # Stack losses into a tensor of shape (batch,)
    return torch.stack(losses)

def run_test():
    print(f"{'='*60}")
    print("Testing Masked SoftDTW (CPU vs GPU)")
    print(f"{'='*60}")
    
    # Parameters
    batch_size = 32
    max_seq_len = 64
    dims = 72
    gamma = 1.0
    
    # Setup Data
    # Random sequence lengths for 'a' (between 10 and max_seq_len)
    seq_lens = torch.randint(low=10, high=max_seq_len + 1, size=(batch_size,))
    # Ensure at least one is max length
    seq_lens[0] = max_seq_len 
    
    # Inputs
    # a: Reference (will be masked by seq_lens)
    # b: Generated (will be used fully)
    a = torch.randn(batch_size, max_seq_len, dims)
    b = torch.randn(batch_size, max_seq_len, dims)
    
    # --- CPU Test ---
    print("\nRunning on CPU...")
    a_cpu = a.clone().requires_grad_(True)
    b_cpu = b.clone().requires_grad_(True)
    seq_lens_cpu = seq_lens.clone()
    
    # Initialize SoftDTW for CPU
    # normalize=False to avoid internal batching issues with unequal lengths
    sdtw_cpu = SoftDTW(use_cuda=False, gamma=gamma, normalize=False)
    
    start_time = time.time()
    loss_cpu = masked_soft_dtw_impl(sdtw_cpu, a_cpu, b_cpu, seq_lens_cpu, normalize=True)
    loss_cpu_mean = loss_cpu.mean()
    loss_cpu_mean.backward()
    cpu_time = time.time() - start_time
    
    print(f"CPU Time: {cpu_time:.4f}s")
    print(f"CPU Loss: {loss_cpu_mean.item():.6f}")
    
    # --- GPU Test ---
    if torch.cuda.is_available():
        print("\nRunning on GPU...")
        a_gpu = a.clone().cuda().requires_grad_(True)
        b_gpu = b.clone().cuda().requires_grad_(True)
        seq_lens_gpu = seq_lens.clone().cuda()
        
        # Initialize SoftDTW for GPU
        sdtw_gpu = SoftDTW(use_cuda=True, gamma=gamma, normalize=False)
        
        # Warmup
        print("Warming up GPU...")
        _ = masked_soft_dtw_impl(sdtw_gpu, a_gpu, b_gpu, seq_lens_gpu, normalize=True)
        torch.cuda.synchronize()
        
        start_time = time.time()
        loss_gpu = masked_soft_dtw_impl(sdtw_gpu, a_gpu, b_gpu, seq_lens_gpu, normalize=True)
        loss_gpu_mean = loss_gpu.mean()
        loss_gpu_mean.backward()
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU Time: {gpu_time:.4f}s")
        print(f"GPU Loss: {loss_gpu_mean.item():.6f}")
        
        # Comparison
        print("\n--- Comparison ---")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
        diff_loss = torch.abs(loss_cpu_mean - loss_gpu_mean.cpu()).item()
        print(f"Loss Difference: {diff_loss:.2e}")
        
        diff_grad_a = torch.abs(a_cpu.grad - a_gpu.grad.cpu()).max().item()
        print(f"Grad A Max Diff: {diff_grad_a:.2e}")
        
        diff_grad_b = torch.abs(b_cpu.grad - b_gpu.grad.cpu()).max().item()
        print(f"Grad B Max Diff: {diff_grad_b:.2e}")
        
        if diff_loss < 1e-4 and diff_grad_a < 1e-4:
            print(">> Status: MATCH")
        else:
            print(">> Status: MISMATCH")
            
    else:
        print("\nCUDA not available, skipping GPU test.")

if __name__ == "__main__":
    run_test()
