import torch
import time
import sys
import os
import warnings
import numba

# Suppress all warnings related to Numba
warnings.simplefilter('ignore', numba.NumbaWarning)

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.soft_dtw_cuda import SoftDTW

def masked_soft_dtw_impl(soft_dtw_module, a, b, seqlen):
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
        loss = soft_dtw_module(a_valid, b_full)
        losses.append(loss.squeeze())

    # Stack losses into a tensor of shape (batch,)
    return torch.stack(losses)

def run_test():
    print(f"{'='*60}")
    print("Testing Masked SoftDTW (CPU vs GPU)")
    print(f"{'='*60}")
    
    # Parameters
    batch_size = 16
    max_seq_len = 240
    dims = 69
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
    sdtw_cpu = SoftDTW(use_cuda=False, gamma=gamma, normalize=True)
    
    start_time = time.time()
    loss_cpu = masked_soft_dtw_impl(sdtw_cpu, a_cpu, b_cpu, seq_lens_cpu)
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
        sdtw_gpu = SoftDTW(use_cuda=True, gamma=gamma, normalize=True)
        
        # Warmup
        print("Warming up GPU...")
        _ = masked_soft_dtw_impl(sdtw_gpu, a_gpu, b_gpu, seq_lens_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        loss_gpu = masked_soft_dtw_impl(sdtw_gpu, a_gpu, b_gpu, seq_lens_gpu)
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
            
        # --- pysdtw Comparison ---
        try:
            from pysdtw import SoftDTW as SoftDTW_Ref
            print("\nRunning with pysdtw (Reference)...")
            
            a_ref = a.clone().cuda().requires_grad_(True)
            b_ref = b.clone().cuda().requires_grad_(True)
            seq_lens_ref = seq_lens.clone().cuda()
            
            # Prepare pysdtw instances
            # We handle normalization manually for unequal lengths to avoid potential library limitations
            sdtw_ref_raw = SoftDTW_Ref(gamma=gamma)
            
            def sdtw_ref_wrapper(x, y):
                xy = sdtw_ref_raw(x, y)
                xx = sdtw_ref_raw(x, x)
                yy = sdtw_ref_raw(y, y)
                return xy - 0.5 * (xx + yy)

            # Warmup
            _ = masked_soft_dtw_impl(sdtw_ref_wrapper, a_ref, b_ref, seq_lens_ref)
            torch.cuda.synchronize()
            
            start_time = time.time()
            loss_ref = masked_soft_dtw_impl(sdtw_ref_wrapper, a_ref, b_ref, seq_lens_ref)
            loss_ref_mean = loss_ref.mean()
            loss_ref_mean.backward()
            torch.cuda.synchronize()
            ref_time = time.time() - start_time
            
            print(f"pysdtw Time: {ref_time:.4f}s")
            print(f"pysdtw Loss: {loss_ref_mean.item():.6f}")
            print(f"Speedup (Custom/pysdtw): {ref_time / gpu_time:.2f}x")
            print(f"Loss Diff: {torch.abs(loss_gpu_mean - loss_ref_mean).item():.2e}")
            diff_grad_a_ref = torch.abs(a_gpu.grad - a_ref.grad).max().item()
            print(f"Grad A Max Diff (Custom vs pysdtw): {diff_grad_a_ref:.2e}")

            diff_grad_b_ref = torch.abs(b_gpu.grad - b_ref.grad).max().item()
            print(f"Grad B Max Diff (Custom vs pysdtw): {diff_grad_b_ref:.2e}")

            diff_grad_a_ref_cpu = torch.abs(a_cpu.grad - a_ref.grad.cpu()).max().item()
            print(f"Grad A Max Diff (CPU vs pysdtw):    {diff_grad_a_ref_cpu:.2e}")

            diff_grad_b_ref_cpu = torch.abs(b_cpu.grad - b_ref.grad.cpu()).max().item()
            print(f"Grad B Max Diff (CPU vs pysdtw):    {diff_grad_b_ref_cpu:.2e}")

            # --- pysdtw CPU Comparison ---
            print("\nRunning with pysdtw (CPU Reference)...")
            a_ref_cpu = a.clone().requires_grad_(True)
            b_ref_cpu = b.clone().requires_grad_(True)
            seq_lens_ref_cpu = seq_lens.clone()

            # Prepare pysdtw instances for CPU
            sdtw_ref_norm_cpu = SoftDTW_Ref(gamma=gamma,  use_cuda=False)
            sdtw_ref_raw_cpu = SoftDTW_Ref(gamma=gamma,  use_cuda=False)

            def sdtw_ref_wrapper_cpu(x, y):
                xy = sdtw_ref_raw_cpu(x, y)
                xx = sdtw_ref_raw_cpu(x, x)
                yy = sdtw_ref_raw_cpu(y, y)
                return xy - 0.5 * (xx + yy)

            # Warmup
            _ = masked_soft_dtw_impl(sdtw_ref_wrapper_cpu, a_ref_cpu, b_ref_cpu, seq_lens_ref_cpu)

            start_time = time.time()
            loss_ref_cpu = masked_soft_dtw_impl(sdtw_ref_wrapper_cpu, a_ref_cpu, b_ref_cpu, seq_lens_ref_cpu)
            loss_ref_cpu_mean = loss_ref_cpu.mean()
            loss_ref_cpu_mean.backward()
            ref_cpu_time = time.time() - start_time

            print(f"pysdtw CPU Time: {ref_cpu_time:.4f}s")
            print(f"pysdtw CPU Loss: {loss_ref_cpu_mean.item():.6f}")
            print(f"Speedup (Custom CPU/pysdtw CPU): {ref_cpu_time / cpu_time:.2f}x")
            
            diff_loss_cpu_ref = torch.abs(loss_cpu_mean - loss_ref_cpu_mean).item()
            print(f"Loss Diff (Custom CPU vs pysdtw CPU): {diff_loss_cpu_ref:.2e}")

            diff_grad_a_ref_cpu_vs_cpu = torch.abs(a_cpu.grad - a_ref_cpu.grad).max().item()
            print(f"Grad A Max Diff (Custom CPU vs pysdtw CPU): {diff_grad_a_ref_cpu_vs_cpu:.2e}")

            # --- pysdtw CPU Comparison ---
            print("\nRunning with pysdtw (CPU Reference)...")
            a_ref_cpu = a.clone().requires_grad_(True)
            b_ref_cpu = b.clone().requires_grad_(True)
            seq_lens_ref_cpu = seq_lens.clone()

            # Prepare pysdtw instances for CPU
            sdtw_ref_norm_cpu = SoftDTW_Ref(gamma=gamma, use_cuda=False)
            sdtw_ref_raw_cpu = SoftDTW_Ref(gamma=gamma,use_cuda=False)

            def sdtw_ref_wrapper_cpu(x, y):
                xy = sdtw_ref_raw_cpu(x, y)
                xx = sdtw_ref_raw_cpu(x, x)
                yy = sdtw_ref_raw_cpu(y, y)
                return xy - 0.5 * (xx + yy)

            # Warmup
            _ = masked_soft_dtw_impl(sdtw_ref_wrapper_cpu, a_ref_cpu, b_ref_cpu, seq_lens_ref_cpu)

            start_time = time.time()
            loss_ref_cpu = masked_soft_dtw_impl(sdtw_ref_wrapper_cpu, a_ref_cpu, b_ref_cpu, seq_lens_ref_cpu)
            loss_ref_cpu_mean = loss_ref_cpu.mean()
            loss_ref_cpu_mean.backward()
            ref_cpu_time = time.time() - start_time

            print(f"pysdtw CPU Time: {ref_cpu_time:.4f}s")
            print(f"pysdtw CPU Loss: {loss_ref_cpu_mean.item():.6f}")
            print(f"Speedup (Custom CPU/pysdtw CPU): {ref_cpu_time / cpu_time:.2f}x")
            
            diff_loss_cpu_ref = torch.abs(loss_cpu_mean - loss_ref_cpu_mean).item()
            print(f"Loss Diff (Custom CPU vs pysdtw CPU): {diff_loss_cpu_ref:.2e}")

            diff_grad_a_ref_cpu_vs_cpu = torch.abs(a_cpu.grad - a_ref_cpu.grad).max().item()
            print(f"Grad A Max Diff (Custom CPU vs pysdtw CPU): {diff_grad_a_ref_cpu_vs_cpu:.2e}")

            diff_grad_b_ref_cpu_vs_cpu = torch.abs(b_cpu.grad - b_ref_cpu.grad).max().item()
            print(f"Grad B Max Diff (Custom CPU vs pysdtw CPU): {diff_grad_b_ref_cpu_vs_cpu:.2e}")

        except ImportError:
            print("\npysdtw not installed. Skipping reference comparison.")

    else:
        print("\nCUDA not available, skipping GPU test.")

if __name__ == "__main__":
    run_test()
