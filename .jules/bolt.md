## 2026-02-27 - [Speaker Embedding Optimization]
**Learning:** Speaker embedding extraction (STFT + ECAPA-TDNN) was being called redundantly for every segment of synthesized text in VoiceCloneNode. Additionally, the STFT was running on CPU because the waveform wasn't moved to the device before spectrogram computation.
**Action:** Pre-calculate voice clone prompts outside the segment loop. Move waveform to device before STFT to leverage GPU acceleration.

## 2026-02-27 - [MPS Precision Bug]
**Learning:** A redundant precision assignment was overwriting Mac-specific optimizations, forcing MPS to use float32, which is significantly slower than float16 on Metal.
**Action:** Remove redundant dtype assignments to preserve device-specific precision optimizations.
