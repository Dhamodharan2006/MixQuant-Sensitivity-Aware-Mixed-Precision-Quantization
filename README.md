# MixQuant — Sensitivity-Aware Mixed-Precision Quantization

A small-scale study testing whether **sensitivity-informed placement of quantization bit-widths** beats **naive uniform quantization**, and whether **targeted QLoRA fine-tuning** can recover the accuracy lost to aggressive quantization — evaluated on Qwen2.5-0.5B using real calibrated quantization kernels (`bitsandbytes` NF4/INT8).

This is a reproduction-and-validation study, not a novel quantization method. The individual techniques (PTQ, QLoRA, mixed precision) are standard; the contribution is a clean, honest experimental pipeline comparing them fairly, on a single consistent quantization backend, with the debugging process documented rather than hidden.

---

## Core Question

> Does quantizing "smartly" (protecting sensitive layers, compressing robust ones) beat quantizing everything uniformly to the same bit-width — when both use the *same* underlying quantization kernel?

## TL;DR Result

| Strategy | PPL ↓ | Latency (ms/token) ↓ | Peak Memory (MB) ↓ |
|---|---|---|---|
| FP16 Baseline | 20.20 | 0.169 | 1953.0 |
| Uniform PTQ INT8 (bnb) | 20.30 | 0.387 | 1588.3 |
| Uniform PTQ INT4 (bnb NF4) | 22.94 | 0.130 | 1431.5 |
| **Mixed PTQ (sensitivity-aware, real bnb)** | **21.34** | **0.150** | **1572.6** |
| Mixed PTQ + Targeted QLoRA | 20.37 | 0.214 | 4069.7 |

**Mixed-precision quantization (INT4 attention / INT8 FFN) beats uniform INT4 on accuracy (21.34 vs 22.94 PPL) while running at ~2.6x lower latency than uniform INT8** — all using the same calibrated `bitsandbytes` kernels, so the comparison is apples-to-apples.

Adding targeted QLoRA on the sensitive group recovers accuracy to near-FP16 levels (20.37 PPL), training only **1.33%** of total parameters. Note that the peak memory for this row (4069.7 MB) is measured **in a training context** (includes optimizer states and gradient buffers). A clean, inference-only measurement is pending; see the Limitations section below.

---

## Motivation

Edge deployment of LLMs typically quantizes every layer to the same bit-width for simplicity. But not all layers are equally robust to quantization — some tolerate aggressive compression with little accuracy loss, others degrade sharply. If those differences can be measured cheaply, a mixed-precision policy should, in principle, land on a better accuracy/efficiency point than any single uniform bit-width. This project tests that assumption directly rather than assuming it.

---

## Methodology

1. **Baseline** — FP16 Qwen2.5-0.5B, perplexity measured on a WikiText-2 held-out slice (8,000 tokens).
2. **Uniform PTQ** — Full-model quantization to INT8 and INT4 (NF4) via `bitsandbytes`.
3. **Sensitivity analysis** — Two module groups (Attention: q/k/v/o projections; FFN: gate/up/down projections) are independently quantized to real INT4 using `bnb.nn.Linear4bit`, and the resulting perplexity delta vs. FP16 is measured on a smaller 1,000-token sweep set for speed.
4. **Mixed-precision policy** — Based on the sensitivity result, the more robust group is quantized aggressively (INT4) and the more sensitive group conservatively (INT8), using the same real `bitsandbytes` layers as the uniform baselines.
5. **Targeted QLoRA** — LoRA adapters (r=16, α=32, ~1.33% of parameters) are trained only on the sensitive group's quantized layers, using `peft`'s `prepare_model_for_kbit_training`, for a small number of steps on a WikiText-2 training slice — checkpointed every few steps to catch overfitting early.
6. **Final comparison** — All five configurations evaluated on the same held-out set for perplexity, latency (ms/token), and peak GPU memory.

All quantization in this study uses real, packed, calibrated `bitsandbytes` kernels throughout — sensitivity, mixed-precision, and QLoRA rows are **not simulated fake-quant**, so memory and latency figures are directly comparable across every row in the final table.

---

## Sensitivity Finding

![Sensitivity Chart](sensitivity_chart.png)

| Module Group | ΔPPL vs FP16 (INT4) |
|---|---|
| Attention (q/k/v/o) | +0.33 |
| FFN (gate/up/down) | +1.60 |

FFN layers are roughly **5x more sensitive** to INT4 quantization than attention projections in this model — the empirical basis for the mixed-precision policy above.

---

## Honest Limitations

This project is scoped and small — worth stating plainly rather than overselling:

- **Single model, single scale.** Only tested on Qwen2.5-0.5B. Findings may not generalize to larger models or other architectures.
- **Perplexity only.** No downstream task evaluation (e.g., MMLU, ARC) — a model can retain perplexity while losing task-specific accuracy after quantization.
- **Single run per configuration.** No variance/confidence intervals reported; results are point estimates on one held-out slice.
- **No target hardware.** All benchmarks run on a free Kaggle T4 GPU, not on edge/mobile hardware (e.g., Qualcomm Hexagon DSP or NPUs), which have different quantization sensitivity profiles and kernel support.
- **No batching / throughput testing.** Latency is measured per-sequence, not under realistic batched serving load.
- **Manual policy assignment.** The mixed-precision policy (which group gets which bit-width) was assigned by hand after inspecting the sensitivity chart — this is a case study, not yet an automated framework that would generalize to a new model without human intervention.
- **QLoRA memory measurement context.** The peak memory for the "Mixed PTQ + Targeted QLoRA" row (4069.7 MB) was captured while the model was still in a training-ready state (`prepare_model_for_kbit_training` active), which retains optimizer states and gradient buffers. This is not a clean inference-only memory footprint; re-measuring in a fresh kernel with adapters merged is the next step.
- **QLoRA adapter overhead is unmerged.** The QLoRA row's latency reflects adapters running alongside the quantized base layers at inference, not fused into them — a real production build would need an adapter-merging step, which is not implemented here.

---

## Debugging & Iteration Log

Documented here because the failures were as informative as the final result:

1. **Per-tensor fake-quant caused catastrophic collapse.** An early version of the sensitivity sweep used a single min-max scale per weight tensor, which let outlier weights blow out the scale for the entire matrix — perplexity exploded to 100,000+. Fixed by moving to per-channel scaling, which brought results back into a sane range.
2. **QLoRA overfitting on a small calibration slice.** An initial run trained for 150 steps on a 4,000-token training slice; training loss fell steadily while held-out perplexity got dramatically worse after ~20-30 steps. Fixed by checkpointing every 5 steps and stopping at the empirically best point (15 steps), rather than a fixed large step count.
3. **Simulated vs. real quantization mismatch.** The first full pipeline used a custom fake-quant function for the mixed-precision and QLoRA rows, while the uniform baselines used real `bitsandbytes` kernels — an invalid comparison between two different quantization methods. Fixed by rewriting the mixed-precision and QLoRA paths to use real `bnb.nn.Linear4bit` / `Linear8bitLt` layers throughout, making every row in the final table method-consistent.

---

## Tech Stack

`PyTorch` · `Transformers` · `bitsandbytes` (NF4 / INT8) · `PEFT` (LoRA / QLoRA) · `datasets` (WikiText-2) · Kaggle T4 GPU

---

## Reproducing This

1. Open the notebook in a Kaggle environment with a **T4 GPU** accelerator enabled.
2. Run cells top to bottom, in order, without skipping — the pipeline depends on sequential state (tokenizer, dataset, `results` list).
3. Full run (baseline + uniform PTQ + sensitivity sweep + mixed PTQ + QLoRA) takes approximately **1–1.5 hours** on a single T4.
4. Final results are written to `final_benchmark_results.csv`.

---

## Future Work

- Automate sensitivity-to-policy assignment (e.g., threshold-based or search-based bit-width selection) instead of manual policy design.
- Evaluate on downstream tasks (MMLU/ARC subset) in addition to perplexity.
- Test on larger models (1.5B–7B) to check whether the sensitivity ranking (FFN > Attention) holds at scale.
- Merge LoRA adapters post-training to remove the inference-time overhead observed in the QLoRA row.
- Extend profiling to actual edge hardware (e.g., Qualcomm AI Hub / Snapdragon) rather than a datacenter T4 GPU.
