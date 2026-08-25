"""Run a compiled DeepScaleLM update across a forced DDP graph split.

Launch with::

    torchrun --standalone --nproc-per-node=2 tests/gpu/ddp_compile_deepscale.py

The tiny bucket is intentional: it makes TorchDynamo's DDP optimizer split the
forward graph around residual coefficients, matching the boundary exercised by
the full modern fine-tuning model without allocating that model in a test.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from a2v2.model import TransformerStack


def main() -> int:
    """Compile, synchronize, and optimize one real DeepScaleLM stack update."""

    if not torch.cuda.is_available():
        raise RuntimeError("compiled DDP DeepScaleLM probe requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    try:
        torch.manual_seed(1_907)
        torch.cuda.manual_seed_all(1_908)
        stack = TransformerStack(
            256,
            8,
            depth=4,
            mlp_ratio=2.0,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            post_mlp_dropout=0.0,
            layerdrop=0.0,
            input_dropout=0.0,
            position_encoding="rope",
            attention_backend="flash",
            ffn_type="geglu",
            initialization="deepscale_lm",
            total_depth=4,
        ).to(device=device, dtype=torch.float16).train()
        state_keys = tuple(stack.state_dict())
        stack.compile(
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=True,
        )
        wrapped = DistributedDataParallel(
            stack,
            device_ids=[local_rank],
            output_device=local_rank,
            forward_sync_buffers=False,
            bucket_cap_mb_list=[1],
        )
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-3)
        value = torch.randn(
            2,
            64,
            256,
            device=device,
            dtype=torch.float16,
        )
        position_ids = torch.arange(64, device=device).expand(2, -1)

        optimizer.zero_grad(set_to_none=True)
        output, targets = wrapped(value, position_ids=position_ids)
        loss = output.float().square().mean() + sum(
            target.float().square().mean() for target in targets
        )
        loss.backward()
        optimizer.step()

        finite = torch.tensor(
            int(torch.isfinite(loss)),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.item()):
            raise RuntimeError("compiled DDP DeepScaleLM update was non-finite")
        if tuple(stack.state_dict()) != state_keys:
            raise RuntimeError("compilation changed DeepScaleLM checkpoint keys")
        report = {
            "rank": rank,
            "world_size": dist.get_world_size(),
            "loss": float(loss.detach()),
            "pass": True,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        dist.barrier(device_ids=[local_rank])
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
