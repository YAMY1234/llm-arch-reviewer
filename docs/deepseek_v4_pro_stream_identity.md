# DeepSeek V4 Pro SGLang stream-identity investigation

Issue #5 was prompted by a graph-on decode step that exposes 65 physical CUDA
stream IDs in a 9.393 ms formal interval even though no more than four streams
are active concurrently. This document records why the Viewer now treats those
IDs as provenance rather than as mandatory rows.

## Evidence

- The accepted profile is
  [`cg_decode_gbs001_8k1k.yaml`](../catalog/deepseek_v4_pro/profiles/tp8/sglang_71de97b_dsv4pro0813_tp8/cg_decode_gbs001_8k1k.yaml).
  It is a CUDA Graph decode capture from rank 1 and identifies job 3424801.
- The pinned SGLang source commit is `71de97b264b04dcd514cf904003028aefe9775c8`.
  In `python/sglang/srt/models/deepseek_v4.py`, the model creates five alternate
  streams. Three serve the KV/compressor/indexer work and two serve additional
  indexer work; the multi-stream path is gated on capture mode.
- The graph-off eager trace for the same implementation and rank contains 2,701
  kernels on two physical stream IDs: 2,688 kernels on stream 23 and 13 on
  stream 59.
- The accepted graph-on formal replay contains 2,675 kernels on 65 physical
  stream IDs. The captured graph was replayed twice in the raw evidence; all
  2,675 graph node IDs map to exactly the same stream ID in both replays.
- Packing the union of actual kernel activity intervals gives an exact peak
  concurrency of four. The 65 rows therefore describe replay-level physical
  identities, not 65 simultaneously active logical lanes.

The raw trace does not expose a complete native application-stream-handle to
replay-stream-ID relation. It is therefore safe to conclude that the 65 IDs are
stable CUDA Graph replay identities, but the exact CUDA-internal allocation
mechanism is intentionally recorded as an evidence-backed inference rather
than as a proven source-level fact.

## Viewer contract

The default **compact activity lanes** mode:

1. keeps every kernel timestamp, duration, physical stream ID, IR target, and
   ownership record unchanged;
2. forms per-physical-stream activity segments from the union of real kernel
   intervals;
3. packs segments into the minimum non-overlapping presentation lanes within a
   reliable role family;
4. pins main/critical compute to the first compute lane while allowing other
   compute segments to reuse its inactive gaps;
5. keeps reliable collective, copy, and artifact-authored semantic roles in
   separate families;
6. retains overlapping kernels within a lane as multiple kernel sublanes, so
   PDL and other same-stream overlap remains visible; and
7. derives its endpoint tolerance from trace numeric resolution and floating
   point precision, rather than from a model-specific constant.

The **physical streams** mode remains the exact debug representation. Compact
lane tooltips list their original stream IDs, and clicking a compact lane label
expands the timeline into physical rows around a representative kernel.

For the accepted DeepSeek V4 Pro step, the expected summary is:

```text
65 physical streams -> 4 compact lanes; peak concurrency 4
```

The compact and physical modes must preserve the same event count, active GPU
union, residency sum, overlap, idle/gap, timestamps, and IR attribution.
