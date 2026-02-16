#!/usr/bin/env python3
"""
PyTorch NVSHMEM 双 rank 示例：一个 rank 使用 put 放入数据，另一个 rank 使用 get 取数据。

运行方式（需要 2 张 GPU 且支持 NVSHMEM）:
  torchrun --nproc_per_node=2 nvshmem_put_get_example.py

依赖: PyTorch 编译时启用 NVSHMEM，且环境已安装 NVSHMEM。
"""

import os
import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def is_nvshmem_available():
    try:
        return symm_mem.is_nvshmem_available()
    except Exception:
        return False


def main():
    # 使用 torchrun 时由环境变量提供
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 2))

    if world_size != 2:
        raise RuntimeError("本示例仅支持 2 个 rank，请使用: torchrun --nproc_per_node=2 nvshmem_put_get_example.py")

    # 初始化进程组（NVSHMEM 通常与 NCCL 一起使用）
    dist.init_process_group(backend="nccl")
    assert dist.get_world_size() == 2

    if not is_nvshmem_available():
        print(f"[Rank {rank}] NVSHMEM 不可用，退出")
        dist.destroy_process_group()
        return

    # 设置 NVSHMEM 为对称内存后端
    symm_mem.set_backend("NVSHMEM")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    # NVSHMEM alloc 内部使用默认进程组 "0"，必须先注册 group info，否则 empty() 会报错
    group_name = dist.group.WORLD.group_name
    with warnings.catch_warnings(action="ignore", category=FutureWarning):
        symm_mem.enable_symm_mem_for_group(group_name)
    dtype = torch.float32
    numel = 1024

    # -------------------------------------------------------------------------
    # 阶段 1: Rank 0 put 数据到 Rank 1
    # -------------------------------------------------------------------------
    # 对称分配：每个 rank 上都有相同布局的 buffer
    tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(tensor, group=group_name)
    signal_pad = hdl.get_signal_pad(rank)
    signal_val = 42

    if rank == 0:
        # Rank 0: 填入数据并 put 到 Rank 1
        tensor.fill_(rank)  # 例如填 0
        torch.ops.symm_mem.nvshmem_put_with_signal(
            tensor, signal_pad, signal_val, 1  # 1 = 目标 rank
        )
        print(f"[Rank 0] PUT 完成: 已将 {numel} 个元素 put 到 Rank 1")
    else:
        # Rank 1: 等待 put 完成后再读本地 buffer（put 写的是本 rank 的对称内存）
        torch.ops.symm_mem.nvshmem_wait_for_signal(signal_pad, signal_val, 0)
        expected = torch.zeros(numel, dtype=dtype, device=device)
        if torch.allclose(tensor, expected):
            print(f"[Rank 1] PUT 接收成功: 收到来自 Rank 0 的数据，校验一致")
        else:
            print(f"[Rank 1] PUT 接收异常: 数据校验失败")

    dist.barrier()

    # -------------------------------------------------------------------------
    # 阶段 2: Rank 1 get 数据从 Rank 0
    # -------------------------------------------------------------------------
    # 重新使用同一块对称内存：Rank 0 写入新数据，Rank 1 从 Rank 0 get
    if rank == 0:
        tensor.fill_(123.0)  # Rank 0 写入要发送的值
    else:
        tensor.fill_(-1.0)   # Rank 1 先清成无效值，等待 get 覆盖

    dist.barrier()

    if rank == 1:
        # Rank 1: 从 Rank 0 get 数据到本地 tensor
        torch.ops.symm_mem.nvshmem_get(tensor, 0)  # 0 = 源 rank
        expected = torch.full((numel,), 123.0, dtype=dtype, device=device)
        if torch.allclose(tensor, expected):
            print(f"[Rank 1] GET 成功: 已从 Rank 0 get 数据，校验一致 (value=123.0)")
        else:
            print(f"[Rank 1] GET 异常: 数据校验失败")

    # 同步：get 无 wait_signal，用 barrier 保证 Rank 0 在 Rank 1 读完后再继续
    dist.barrier()

    print(f"[Rank {rank}] 示例结束")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
