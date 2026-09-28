# Compare voices locally

Uses file_1.m4a and each nonblank line of test_sentences.txt to compare Qwen3-TTS 0.6B and 1.7B Base voice cloning. No fine-tuning is required.

## Setup (PowerShell)
Install FFmpeg on PATH and create a Python 3.12 environment:
```powershell
conda create -n qwen-compare python=3.12 -y
conda activate qwen-compare
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-inference.txt
python compare_models.py
```

Weights download on first use. The recording stays local. The script runs models sequentially and saves WAV files plus a compare.html listening page under a new outputs/comparison_* folder. Open that HTML file in your browser.

By default the reference is the first 7.3 seconds, intended to capture the first two sentences in the supplied recording. Listen to reference.wav and check reference.txt: the automatic transcript may contain errors. To use a corrected transcript:
```powershell
python compare_models.py --ref-text reference_transcript.txt
```
That text must match the selected excerpt, not the entire five-minute recording. Use --start and --duration to select another clean section ending at a sentence boundary.

To run just one model:
```powershell
python compare_models.py --models 0.6B
python compare_models.py --models 1.7B
```
The 1.7B model may exceed available GPU memory. CPU fallback is explicit and slower:
```powershell
python compare_models.py --models 1.7B --device cpu
```

These scripts have been syntax-checked, not run through full generation in this folder. Compare the same sentences and judge accent, speaker similarity, pronunciation, and naturalness. Matching seeds do not make the two different architectures generate identically.

Official model documentation: https://github.com/QwenLM/Qwen3-TTS

