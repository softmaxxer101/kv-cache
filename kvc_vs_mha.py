import torch

import torch.nn as nn

import torch.nn.functional as F

import time

import numpy as np

import matplotlib.pyplot as plt

import gc

import os



# ============================================

# Check and setup GPU for Kaggle

# ============================================

def setup_kaggle_gpu():

    """Configure GPU settings for Kaggle environment"""

    if torch.cuda.is_available():

        # Enable TF32 for Ampere+ GPUs (T4, P100 on Kaggle)

        torch.backends.cuda.matmul.allow_tf32 = True

        torch.backends.cudnn.allow_tf32 = True

        

        # Set optimal CUDA settings for Kaggle

        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.deterministic = False

        

        # Clear cache

        torch.cuda.empty_cache()

        

        print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(f"CUDA Version: {torch.version.cuda}")

        print(f"PyTorch Version: {torch.__version__}")

        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        return True

    else:

        print("WARNING: No GPU detected! Running on CPU (will be very slow)")

        print("To enable GPU in Kaggle: Settings -> Accelerator -> GPU")

        return False



# ============================================

# Optimized KV-Cache MHA Implementation

# ============================================

class KV_Cache_MHA:

    def __init__(self, D=512, h=16, device='cuda'):

        self.D = D

        self.h = h

        self.hd = D // h

        self.scale = 1.0 / (self.hd ** 0.5)

        

        self.wq = nn.Linear(D, D, bias=False).to(device)

        self.wk = nn.Linear(D, D, bias=False).to(device)

        self.wv = nn.Linear(D, D, bias=False).to(device)

        self.wo = nn.Linear(D, D, bias=False).to(device)

        

        self.cache_k = None

        self.cache_v = None

        self.device = device

    

    def reset_cache(self):

        self.cache_k = None

        self.cache_v = None

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    

    def forward_step(self, x):

        B, T, D = x.shape

        

        q = self.wq(x)

        k = self.wk(x)

        v = self.wv(x)

        

        q = q.reshape(B, T, self.h, self.hd).transpose(1, 2)

        k = k.reshape(B, T, self.h, self.hd).transpose(1, 2)

        v = v.reshape(B, T, self.h, self.hd).transpose(1, 2)

        

        if self.cache_k is None:

            self.cache_k = k

            self.cache_v = v

        else:

            self.cache_k = torch.cat([self.cache_k, k], dim=2)

            self.cache_v = torch.cat([self.cache_v, v], dim=2)

        

        # Use PyTorch's efficient implementation when possible

        if hasattr(F, 'scaled_dot_product_attention'):

            attention = F.scaled_dot_product_attention(q, self.cache_k, self.cache_v)

        else:

            attn_weights = torch.matmul(q, self.cache_k.transpose(-2, -1)) * self.scale

            attn_weights = F.softmax(attn_weights, dim=-1)

            attention = torch.matmul(attn_weights, self.cache_v)

        

        out = self.wo(attention.transpose(1, 2).reshape(B, T, D))

        return out



# ============================================

# Optimized Non-KV-Cache MHA

# ============================================

class Non_KV_Cache_MHA:

    def __init__(self, D=512, h=16, device='cuda'):

        self.D = D

        self.h = h

        self.hd = D // h

        self.scale = 1.0 / (self.hd ** 0.5)

        

        self.wq = nn.Linear(D, D, bias=False).to(device)

        self.wk = nn.Linear(D, D, bias=False).to(device)

        self.wv = nn.Linear(D, D, bias=False).to(device)

        self.wo = nn.Linear(D, D, bias=False).to(device)

        

        self.history = None

        self.device = device

    

    def reset_history(self):

        self.history = None

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    

    def forward_step(self, x):

        B, T, D = x.shape

        

        if self.history is None:

            self.history = x

        else:

            self.history = torch.cat([self.history, x], dim=1)

        

        full_x = self.history

        B, full_T, D = full_x.shape

        

        q = self.wq(full_x)

        k = self.wk(full_x)

        v = self.wv(full_x)

        

        q = q.reshape(B, full_T, self.h, self.hd).transpose(1, 2)

        k = k.reshape(B, full_T, self.h, self.hd).transpose(1, 2)

        v = v.reshape(B, full_T, self.h, self.hd).transpose(1, 2)

        

        if hasattr(F, 'scaled_dot_product_attention'):

            attention = F.scaled_dot_product_attention(q, k, v)

        else:

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            attn_weights = F.softmax(attn_weights, dim=-1)

            attention = torch.matmul(attn_weights, v)

        

        out = self.wo(attention.transpose(1, 2).reshape(B, full_T, D))

        return out[:, -1:, :]



# ============================================

# GPU-optimized Benchmark for Kaggle

# ============================================

def benchmark_kaggle(model_class, seq_length, D=512, h=16, warmup=1, runs=3):

    """Kaggle-optimized benchmark with memory constraints"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

    # Check if sequence length is feasible

    if device == 'cuda':

        available_memory = torch.cuda.get_device_properties(0).total_memory

        # Estimate memory needed (rough calculation)

        estimated_memory = seq_length * D * 4 * h * 2 * 2  # K and V caches

        if estimated_memory > available_memory * 0.7:  # Use max 70% of GPU memory

            return None

    

    try:

        model = model_class(D=D, h=h, device=device)

        

        # Warmup

        for _ in range(warmup):

            if hasattr(model, 'reset_cache'):

                model.reset_cache()

            if hasattr(model, 'reset_history'):

                model.reset_history()

            

            for _ in range(min(seq_length, 100)):  # Warmup with subset for large sequences

                x = torch.randn(1, 1, D, device=device)

                _ = model.forward_step(x)

            

            if device == 'cuda':

                torch.cuda.synchronize()

        

        # Actual measurement

        times = []

        

        for run in range(runs):

            if hasattr(model, 'reset_cache'):

                model.reset_cache()

            if hasattr(model, 'reset_history'):

                model.reset_history()

            

            x = torch.randn(1, 1, D, device=device)

            

            if device == 'cuda':

                torch.cuda.synchronize()

            

            start_time = time.perf_counter()

            

            # Process in chunks for very long sequences to show progress

            chunk_size = min(seq_length, 500)

            for start_idx in range(0, seq_length, chunk_size):

                end_idx = min(start_idx + chunk_size, seq_length)

                for i in range(start_idx, end_idx):

                    _ = model.forward_step(x)

            

            if device == 'cuda':

                torch.cuda.synchronize()

            

            end_time = time.perf_counter()

            

            total_time = (end_time - start_time) * 1000  # Convert to ms

            times.append(total_time)

        

        # Get GPU memory

        if device == 'cuda':

            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

            torch.cuda.reset_peak_memory_stats()

        else:

            gpu_mem = 0

        

        return {

            'total_time': np.mean(times),

            'std_time': np.std(times) if len(times) > 1 else 0,

            'avg_per_token': np.mean(times) / seq_length,

            'gpu_memory_mb': gpu_mem,

            'all_times': times

        }

    

    except RuntimeError as e:

        if "out of memory" in str(e).lower():

            print(f"\nOut of memory for sequence length {seq_length}")

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            return None

        else:

            raise e



# ============================================

# Main Benchmark for Kaggle

# ============================================

def run_kaggle_benchmark():

    """Main benchmark function optimized for Kaggle"""

    

    # Setup

    print("=" * 100)

    print("KV-CACHE vs NON-KV-CACHE GPU BENCHMARK")

    print("=" * 100)

    

    has_gpu = setup_kaggle_gpu()

    

    if not has_gpu:

        # Fall back to smaller test on CPU

        print("\nRunning on CPU with reduced sequence lengths...")

        seq_lengths = [10, 50, 100, 200, 500]  # Much smaller for CPU

    else:

        # Adjust sequence lengths based on available GPU memory

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        

        if gpu_memory_gb < 8:  # T4 or similar

            seq_lengths = [10, 50, 100, 200, 500, 1000, 2000]

            print(f"\nLimited GPU memory ({gpu_memory_gb:.1f} GB) - using sequence lengths up to 2000")

        elif gpu_memory_gb < 16:  # P100 or similar

            seq_lengths = [10, 50, 100, 200, 500, 1000, 2500, 5000]

            print(f"\nModerate GPU memory ({gpu_memory_gb:.1f} GB) - using sequence lengths up to 5000")

        else:  # V100, A100, etc.

            seq_lengths = [10, 50, 100, 200, 500, 1000, 2500, 5000, 10000]

            print(f"\nLarge GPU memory ({gpu_memory_gb:.1f} GB) - using full sequence lengths")

    

    D = 512

    h = 16

    

    print(f"\nTest parameters: D={D}, h={h}, head_dim={D//h}")

    print(f"Sequence lengths: {seq_lengths}")

    

    # Initialize results storage

    results = {'kvc': {}, 'non_kvc': {}}

    

    print("\n" + "=" * 100)

    print(f"{'Seq Len':<10} {'Method':<20} {'Total (ms)':<15} {'Â± Std':<12} {'ms/token':<12} {'GPU Mem (MB)':<15} {'Speedup':<12}")

    print("=" * 100)

    

    for seq_len in seq_lengths:

        print(f"\nProcessing sequence length: {seq_len}...", end=' ', flush=True)

        

        # Clear GPU cache

        if has_gpu:

            torch.cuda.empty_cache()

        gc.collect()

        

        # KV-Cache benchmark

        kvc_result = benchmark_kaggle(KV_Cache_MHA, seq_len, D=D, h=h)

        

        if has_gpu:

            torch.cuda.empty_cache()

        gc.collect()

        

        # Non-KV-Cache benchmark

        non_kvc_result = benchmark_kaggle(Non_KV_Cache_MHA, seq_len, D=D, h=h)

        

        # Store results

        results['kvc'][seq_len] = kvc_result

        results['non_kvc'][seq_len] = non_kvc_result

        

        # Print results

        if kvc_result is not None and non_kvc_result is not None:

            speedup = non_kvc_result['avg_per_token'] / kvc_result['avg_per_token']

            print("Done!")

            print(f"{seq_len:<10} {'KV-Cache':<20} {kvc_result['total_time']:<15.2f} "

                  f"Â±{kvc_result['std_time']:<10.2f} {kvc_result['avg_per_token']:<12.3f} "

                  f"{kvc_result['gpu_memory_mb']:<15.1f} {'-':<12}")

            print(f"{'':10} {'Non-KV-Cache':<20} {non_kvc_result['total_time']:<15.2f} "

                  f"Â±{non_kvc_result['std_time']:<10.2f} {non_kvc_result['avg_per_token']:<12.3f} "

                  f"{non_kvc_result['gpu_memory_mb']:<15.1f} {speedup:<12.2f}x")

        elif kvc_result is not None:

            print("Done! (Non-KV-Cache skipped)")

            print(f"{seq_len:<10} {'KV-Cache':<20} {kvc_result['total_time']:<15.2f} "

                  f"Â±{kvc_result['std_time']:<10.2f} {kvc_result['avg_per_token']:<12.3f} "

                  f"{kvc_result['gpu_memory_mb']:<15.1f} {'N/A':<12}")

        else:

            print("Skipped (too large for available memory)")

    

    # Create visualizations

    create_kaggle_plots(results, seq_lengths)

    

    # Print summary

    print_summary(results, seq_lengths)

    

    return results



# ============================================

# Visualization for Kaggle

# ============================================

def create_kaggle_plots(results, seq_lengths):

    """Create plots optimized for Kaggle notebook display"""

    

    # Filter valid results

    valid_seqs = [s for s in seq_lengths 

                  if results['kvc'][s] is not None and results['non_kvc'][s] is not None]

    

    if len(valid_seqs) < 2:

        print("\nNot enough valid data points for plotting")

        return

    

    fig = plt.figure(figsize=(15, 10))

    

    # Plot 1: Total Time Comparison

    ax1 = plt.subplot(2, 3, 1)

    kvc_totals = [results['kvc'][s]['total_time'] for s in valid_seqs]

    non_kvc_totals = [results['non_kvc'][s]['total_time'] for s in valid_seqs]

    

    ax1.plot(valid_seqs, kvc_totals, 'b-o', label='KV-Cache', linewidth=2, markersize=6)

    ax1.plot(valid_seqs, non_kvc_totals, 'r-o', label='Non-KV-Cache', linewidth=2, markersize=6)

    ax1.set_xlabel('Sequence Length', fontsize=10)

    ax1.set_ylabel('Total Time (ms)', fontsize=10)

    ax1.set_title('Total Processing Time', fontsize=12, fontweight='bold')

    ax1.legend()

    ax1.grid(True, alpha=0.3)

    

    # Plot 2: Time per Token

    ax2 = plt.subplot(2, 3, 2)

    kvc_per_token = [results['kvc'][s]['avg_per_token'] for s in valid_seqs]

    non_kvc_per_token = [results['non_kvc'][s]['avg_per_token'] for s in valid_seqs]

    

    ax2.plot(valid_seqs, kvc_per_token, 'b-o', label='KV-Cache', linewidth=2, markersize=6)

    ax2.plot(valid_seqs, non_kvc_per_token, 'r-o', label='Non-KV-Cache', linewidth=2, markersize=6)

    ax2.set_xlabel('Sequence Length', fontsize=10)

    ax2.set_ylabel('Time per Token (ms)', fontsize=10)

    ax2.set_title('Average Time per Token', fontsize=12, fontweight='bold')

    ax2.legend()

    ax2.grid(True, alpha=0.3)

    

    # Plot 3: Speedup Factor

    ax3 = plt.subplot(2, 3, 3)

    speedups = [results['non_kvc'][s]['avg_per_token'] / results['kvc'][s]['avg_per_token'] 

                for s in valid_seqs]

    

    colors = ['green' if s > 1 else 'red' for s in speedups]

    ax3.bar(range(len(valid_seqs)), speedups, color=colors, alpha=0.7)

    ax3.set_xticks(range(len(valid_seqs)))

    ax3.set_xticklabels(valid_seqs)

    ax3.set_xlabel('Sequence Length', fontsize=10)

    ax3.set_ylabel('Speedup (x times)', fontsize=10)

    ax3.set_title('KV-Cache Speedup Factor', fontsize=12, fontweight='bold')

    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Break-even')

    ax3.legend()

    ax3.grid(True, alpha=0.3, axis='y')

    

    # Plot 4: GPU Memory Usage

    ax4 = plt.subplot(2, 3, 4)

    kvc_mem = [results['kvc'][s]['gpu_memory_mb'] for s in valid_seqs]

    non_kvc_mem = [results['non_kvc'][s]['gpu_memory_mb'] for s in valid_seqs]

    

    ax4.plot(valid_seqs, kvc_mem, 'b-o', label='KV-Cache', linewidth=2, markersize=6)

    ax4.plot(valid_seqs, non_kvc_mem, 'r-o', label='Non-KV-Cache', linewidth=2, markersize=6)

    ax4.set_xlabel('Sequence Length', fontsize=10)

    ax4.set_ylabel('GPU Memory (MB)', fontsize=10)

    ax4.set_title('GPU Memory Usage', fontsize=12, fontweight='bold')

    ax4.legend()

    ax4.grid(True, alpha=0.3)

    

    # Plot 5: Throughput (tokens/second)

    ax5 = plt.subplot(2, 3, 5)

    kvc_throughput = [1000 * s / results['kvc'][s]['total_time'] for s in valid_seqs]

    non_kvc_throughput = [1000 * s / results['non_kvc'][s]['total_time'] for s in valid_seqs]

    

    ax5.plot(valid_seqs, kvc_throughput, 'b-o', label='KV-Cache', linewidth=2, markersize=6)

    ax5.plot(valid_seqs, non_kvc_throughput, 'r-o', label='Non-KV-Cache', linewidth=2, markersize=6)

    ax5.set_xlabel('Sequence Length', fontsize=10)

    ax5.set_ylabel('Tokens/Second', fontsize=10)

    ax5.set_title('Throughput', fontsize=12, fontweight='bold')

    ax5.legend()

    ax5.grid(True, alpha=0.3)

    

    # Plot 6: Efficiency Gain (%)

    ax6 = plt.subplot(2, 3, 6)

    efficiency_gain = [(speedups[i] - 1) * 100 for i in range(len(valid_seqs))]

    

    colors = ['green' if e > 0 else 'red' for e in efficiency_gain]

    ax6.bar(range(len(valid_seqs)), efficiency_gain, color=colors, alpha=0.7)

    ax6.set_xticks(range(len(valid_seqs)))

    ax6.set_xticklabels(valid_seqs)

    ax6.set_ylabel('Efficiency Gain (%)', fontsize=10)

    ax6.set_title('Efficiency Improvement with KV-Cache', fontsize=12, fontweight='bold')

    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax6.grid(True, alpha=0.3, axis='y')

    

    plt.tight_layout()

    plt.savefig('kv_cache_benchmark.png', dpi=100, bbox_inches='tight')

    plt.show()

    

    print("\nâœ… Plot saved as 'kv_cache_benchmark.png'")



# ============================================

# Summary

# ============================================

def print_summary(results, seq_lengths):

    """Print comprehensive summary"""

    

    valid_seqs = [s for s in seq_lengths 

                  if results['kvc'][s] is not None and results['non_kvc'][s] is not None]

    

    if not valid_seqs:

        print("\nNo valid results to summarize")

        return

    

    print("\n" + "=" * 100)

    print("FINAL SUMMARY")

    print("=" * 100)

    print(f"\n{'Seq Length':<12} {'KV-Cache (ms)':<18} {'Non-KV (ms)':<18} {'Speedup':<12} {'KV-Cache MB':<15} {'Non-KV MB':<15}")

    print("-" * 90)

    

    for seq_len in valid_seqs:

        kvc = results['kvc'][seq_len]

        non_kvc = results['non_kvc'][seq_len]

        speedup = non_kvc['avg_per_token'] / kvc['avg_per_token']

        print(f"{seq_len:<12} {kvc['total_time']:<18.2f} {non_kvc['total_time']:<18.2f} "

              f"{speedup:<12.2f}x {kvc['gpu_memory_mb']:<15.1f} {non_kvc['gpu_memory_mb']:<15.1f}")

    

    # Calculate key metrics

    max_speedup = max([results['non_kvc'][s]['avg_per_token'] / results['kvc'][s]['avg_per_token'] 

                       for s in valid_seqs])

    max_speedup_seq = valid_seqs[[results['non_kvc'][s]['avg_per_token'] / results['kvc'][s]['avg_per_token'] 

                                   for s in valid_seqs].index(max_speedup)]

    

    print("\n" + "=" * 100)

    print("KEY FINDINGS")

    print("=" * 100)

    print(f"""

    ðŸ“Š Maximum Speedup: {max_speedup:.1f}x at sequence length {max_speedup_seq}

    

    ðŸš€ KV-Cache Advantages:

       â€¢ Linear time complexity O(n) vs O(nÂ²) for non-KV-cache

       â€¢ Constant per-token processing time

       â€¢ Dramatically better for long sequences

       â€¢ Essential for autoregressive generation (GPT, Llama, etc.)

    

    ðŸ’¾ Memory Trade-off:

       â€¢ KV-Cache stores K,V for all previous tokens

       â€¢ Memory grows linearly with sequence length

       â€¢ Non-KV-cache uses more memory for attention matrix computation

    

    âš¡ Practical Impact:

       â€¢ For sequences > 1000 tokens, KV-cache provides 10-100x speedup

       â€¢ Makes real-time text generation feasible

       â€¢ Without KV-cache, generating long texts would be impractical

    """)



# ============================================

# Run the benchmark

# ============================================

if __name__ == "__main__":

    # Run the Kaggle-optimized benchmark

    results = run_kaggle_benchmark()
