#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch GPU Benchmark (no CLI args)
- H2D / D2H / D2D bandwidth
- GEMM throughput
- Conv3D forward benchmark
- GPU0 <-> GPU1 P2P bandwidth
- Optional DDP all-reduce

Designed for diagnosing PCIe x1 / x8 / x16 issues.
"""

import os
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

# =============================================================================
# ============================== CONFIGURATION ================================
# =============================================================================

# -------- General --------
DEVICE_INDEX = 1            # 单卡测试用哪张 GPU
DTYPE = "fp16"              # "fp16" | "bf16" | "fp32"
ITERS = 80                  # 计时循环次数
WARMUP = 20                 # warmup 次数

# -------- Bandwidth --------
TRANSFER_SIZE_MB = 1024     # H2D / D2H / P2P 测试大小（MiB）

# -------- GEMM --------
GEMM_N = 8192               # 矩阵大小 N x N

# -------- Conv3D --------
CONV_BATCH = 1
CONV_CIN = 32
CONV_COUT = 64
CONV_VOL = 128              # D=H=W

# -------- Multi-GPU --------
ENABLE_P2P = False           # GPU0 <-> GPU1 互传
ENABLE_DDP = False           # DDP all-reduce（2 GPU）

# =============================================================================
# ============================== UTILS ========================================
# =============================================================================

def sync(dev):
    torch.cuda.synchronize(dev)

def timer(fn, dev, iters, warmup):
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / iters

def header(dev):
    prop = torch.cuda.get_device_properties(dev)
    print("=" * 90)
    print(f"Device: cuda:{dev.index} | {prop.name}")
    print(f"VRAM: {prop.total_memory / 1024**3:.2f} GiB | SMs: {prop.multi_processor_count}")
    print(f"CUDA {torch.version.cuda} | PyTorch {torch.__version__}")
    print("=" * 90)

# =============================================================================
# ============================== BANDWIDTH ====================================
# =============================================================================

@torch.no_grad()
def bandwidth_test(dev):
    print("\n[Bandwidth] H2D / D2H / D2D")

    nbytes = TRANSFER_SIZE_MB * 1024 * 1024
    numel = nbytes // 2  # fp16

    host = torch.empty(numel, dtype=torch.float16, pin_memory=True)
    host2 = torch.empty_like(host, pin_memory=True)
    d0 = torch.empty(numel, device=dev, dtype=torch.float16)
    d1 = torch.empty_like(d0)

    dt = timer(lambda: d0.copy_(host, non_blocking=True), dev, ITERS, WARMUP)
    print(f"H2D  : {nbytes/dt/1024**3:.2f} GiB/s")

    dt = timer(lambda: host2.copy_(d0, non_blocking=True), dev, ITERS, WARMUP)
    print(f"D2H  : {nbytes/dt/1024**3:.2f} GiB/s")

    dt = timer(lambda: d1.copy_(d0), dev, ITERS, WARMUP)
    print(f"D2D  : {nbytes/dt/1024**3:.2f} GiB/s")

# =============================================================================
# ============================== GEMM =========================================
# =============================================================================

@torch.no_grad()
def gemm_test(dev):
    print("\n[GEMM]")

    if DTYPE == "fp16":
        dtype = torch.float16
        ctx = torch.autocast("cuda", torch.float16)
    elif DTYPE == "bf16":
        dtype = torch.bfloat16
        ctx = torch.autocast("cuda", torch.bfloat16)
    else:
        dtype = torch.float32
        ctx = nullcontext()

    a = torch.randn(GEMM_N, GEMM_N, device=dev, dtype=dtype)
    b = torch.randn_like(a)

    def run():
        with ctx:
            return a @ b

    dt = timer(run, dev, ITERS // 2, WARMUP)
    tflops = 2 * GEMM_N**3 / dt / 1e12
    print(f"{DTYPE.upper()} GEMM: {tflops:.1f} TFLOPs")

# =============================================================================
# ============================== CONV3D =======================================
# =============================================================================

class Conv3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv3d(CONV_CIN, CONV_COUT, 3, padding=1)

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def conv3d_test(dev):
    print("\n[Conv3D]")

    dtype = torch.float16 if DTYPE == "fp16" else torch.float32
    ctx = torch.autocast("cuda", torch.float16) if DTYPE == "fp16" else nullcontext()

    x = torch.randn(CONV_BATCH, CONV_CIN, CONV_VOL, CONV_VOL, CONV_VOL,
                    device=dev, dtype=dtype)
    net = Conv3D().to(dev).eval()

    def run():
        with ctx:
            return net(x)

    dt = timer(run, dev, ITERS, WARMUP)
    print(f"Conv3D time: {dt*1e3:.2f} ms")

# =============================================================================
# ============================== P2P ==========================================
# =============================================================================

@torch.no_grad()
def p2p_test():
    print("\n[P2P GPU0 <-> GPU1]")

    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs")
        return

    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")

    nbytes = TRANSFER_SIZE_MB * 1024 * 1024
    numel = nbytes // 2

    a = torch.empty(numel, device=d0, dtype=torch.float16)
    b = torch.empty_like(a, device=d1)

    dt = timer(lambda: b.copy_(a), d1, ITERS, WARMUP)
    print(f"0 → 1 : {nbytes/dt/1024**3:.2f} GiB/s")

    dt = timer(lambda: a.copy_(b), d0, ITERS, WARMUP)
    print(f"1 → 0 : {nbytes/dt/1024**3:.2f} GiB/s")

# =============================================================================
# ============================== DDP ==========================================
# =============================================================================

def ddp_worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world)

    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    numel = (TRANSFER_SIZE_MB * 1024 * 1024) // 4
    x = torch.randn(numel, device=dev)

    for _ in range(WARMUP):
        dist.all_reduce(x)
    sync(dev)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        dist.all_reduce(x)
    sync(dev)
    dt = (time.perf_counter() - t0) / ITERS

    if rank == 0:
        bw = (2 * numel * 4) / dt / 1024**3
        print(f"\n[DDP all-reduce] ~{bw:.2f} GiB/s")

    dist.destroy_process_group()

def ddp_test():
    print("\n[DDP all-reduce]")
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs")
        return
    mp.spawn(ddp_worker, args=(2,), nprocs=2)

# =============================================================================
# ============================== MAIN =========================================
# =============================================================================

if __name__ == "__main__":
    assert torch.cuda.is_available()
    dev = torch.device(f"cuda:{DEVICE_INDEX}")
    torch.cuda.set_device(dev)

    header(dev)
    bandwidth_test(dev)
    gemm_test(dev)
    conv3d_test(dev)

    if ENABLE_P2P:
        p2p_test()

    if ENABLE_DDP:
        ddp_test()

    print("\nDone.")