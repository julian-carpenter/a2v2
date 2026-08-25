# Regional Pretraining Compilation Design

## Problem

Modern animal2vec2 pretraining removes a deterministic, update-dependent set
of tokens before the student Transformer. Consequently, `MaskInfo.ids_keep`
and the student sequence length vary across otherwise valid batches. Compiling
the complete `Animal2VecPretrainingModel.forward` makes the continuation after
mask construction specialize on those lengths. Long runs retain several large
Inductor graphs, hit TorchDynamo's recompilation limit, and can exhaust a 40 GB
A100 even though a single update fits.

An OOM on one rank is followed by NCCL timeouts on surviving ranks. The
collective timeout is therefore a secondary symptom, not a communication
failure or an effective-batch-size problem.

## Requirements

- Preserve the frozen paper-reproduction execution path and its launcher hash.
- Keep modern pretraining compiled without allowing mask-dependent graph
  accumulation.
- Retain FlashAttention, RoPE, GEGLU, DeepScaleLM, AdaGC, AdamW8bit, CLS, and
  weight-decay annealing behavior.
- Preserve parameter identities, checkpoint keys, optimizer construction order,
  DDP construction order, resume fingerprints, and numerical model semantics.
- Keep modern fine-tuning eligible for complete-model compilation.
- Fail before training when compiled pretraining requests static shapes.
- Make the effective compile scope visible in the terminal training summary.

## Architecture

`_compile_model_in_place` will select its compilation scope from the concrete
model type. Fine-tuning and unrelated modules retain the existing one-region,
complete-model policy. `Animal2VecPretrainingModel` instead keeps its outer
forward eager and compiles every student and EMA-teacher `TransformerBlock` in
place. The block boundary contains the expensive attention and feed-forward
math while excluding waveform mixup, deterministic NumPy mask generation,
`MaskInfo`, masked-token gathering/restoration, teacher target selection, and
masked prediction indexing.

Regional pretraining requires `torch_compile_dynamic: true`. At a block entry,
the retained sequence length is then an input dimension rather than a Python
scalar captured above a graph break. Static or unspecified dynamic-shape policy
is rejected with an actionable error instead of starting a run that may later
accumulate specialized graphs.

The policy returns an immutable report containing `scope` and `regions`. The
training summary records these fields beside the configured backend, mode,
fullgraph, and dynamic settings. Compilation remains in-place before optimizer
and DDP construction, so parameters and checkpoint schemas remain unchanged.

## Compiler Decorators

`compute_mask_indices` and `make_mask_info` no longer execute under a compiled
outer forward. Their `torch.compiler.disable` decorators are therefore
redundant and will be removed. These functions remain ordinary deterministic
eager functions and retain their existing unit coverage.

## Configuration and Launcher

The modern pretraining YAML and `animal2vec2_benchmark.sh` will explicitly set
`common.torch_compile_dynamic=true`. The benchmark profile will record
`pretrain_torch_compile_scope=transformer_blocks` and
`finetune_torch_compile_scope=model`. Fine-tuning keeps complete-model dynamic
compilation. The legacy reproduction launcher and legacy configs are not changed.

## Verification

1. A unit test captures `Module.compile` calls and proves that a pretraining
   model compiles only student and teacher Transformer blocks with exact options.
2. A unit test proves compiled pretraining rejects non-dynamic policy before
   touching any module.
3. Existing generic compile-policy tests continue to prove complete-model
   behavior and construction ordering.
4. Integration tests verify the reported scope and checkpoint state keys.
5. A multi-update modern pretraining regression uses changing retained-token
   lengths without compiling the outer forward.
6. Focused CUDA tests exercise strict FlashAttention and DDP.
7. An eight-A100 resume probe runs past the previous recompilation window with a
   clean Inductor cache and records recompilation logs and peak memory.
8. The frozen reproduction launcher checksum is verified unchanged.
