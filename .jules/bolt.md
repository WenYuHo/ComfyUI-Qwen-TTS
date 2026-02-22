## 2025-05-14 - [Batching Inference Segments in ComfyUI Nodes]
**Learning:** Sequential processing of TTS segments (sentences) is a major bottleneck in GPU-based custom nodes. Even if segments are short, the overhead of multiple model calls and GPU kernel launches accumulates significantly.
**Action:** Always check if the underlying model wrapper supports batching and refactor sequential loops into a single batched call. This typically involves collecting valid inputs, mapping indices to results, and interleaving with non-batched elements like silences.
