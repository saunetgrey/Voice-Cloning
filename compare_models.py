"""Compare Qwen3-TTS Base models using one local voice reference."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parent
MODELS = {s: f"Qwen/Qwen3-TTS-12Hz-{s}-Base" for s in ("0.6B", "1.7B")}


def worker(args):
    import torch
    import soundfile as sf
    from transformers import set_seed
    from qwen_tts import Qwen3TTSModel
    run = Path(args.worker)
    config = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install CUDA PyTorch or use --device cpu.")
    dtype = torch.float32 if device == "cpu" else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = Qwen3TTSModel.from_pretrained(
        MODELS[args.model], device_map=device, dtype=dtype, attn_implementation="sdpa")
    folder = run / args.model
    folder.mkdir(exist_ok=True)
    with torch.inference_mode():
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(run / "reference.wav"),
            ref_text=config["reference_text"], x_vector_only_mode=False)
        for i, text in enumerate(config["sentences"], 1):
            set_seed(config["seed"] + i)
            print(f"{args.model} [{i}/{len(config['sentences'])}]: {text}", flush=True)
            waves, sr = model.generate_voice_clone(
                text=text, language="English", voice_clone_prompt=prompt,
                max_new_tokens=600)
            sf.write(folder / f"{i:02d}.wav", waves[0], sr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=ROOT / "file_1.m4a")
    parser.add_argument("--sentences", type=Path, default=ROOT / "test_sentences.txt")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=7.3)
    parser.add_argument("--ref-text", type=Path, help="UTF-8 transcript of the selected audio excerpt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=list(MODELS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
    if args.worker:
        return worker(args)
    if not args.audio.is_file() or not args.sentences.is_file():
        parser.error("Audio or test sentences file not found.")
    if args.start < 0 or not 1 <= args.duration <= 30:
        parser.error("Use a nonnegative start and a reference duration between 1 and 30 seconds.")
    if not shutil.which("ffmpeg"):
        parser.error("FFmpeg must be installed and available on PATH.")
    sentences = [s.strip() for s in args.sentences.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not sentences:
        parser.error("No test sentences found.")
    import soundfile as sf
    run = ROOT / "outputs" / datetime.now().strftime("comparison_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                    "-ss", str(args.start), "-i", str(args.audio.resolve()),
                    "-t", str(args.duration), "-vn", "-ac", "1", "-ar", "24000",
                    "-c:a", "pcm_s16le", str(run / "reference.wav")], check=True)
    if sf.info(run / "reference.wav").duration < 1:
        raise RuntimeError("Selected excerpt is too short. Check --start and --duration.")
    if args.ref_text:
        reference_text = args.ref_text.read_text(encoding="utf-8-sig").strip()
    else:
        from faster_whisper import WhisperModel
        print("Transcribing the reference excerpt locally...", flush=True)
        recognizer = WhisperModel("small.en", device="cpu", compute_type="int8",
                                  download_root=str(ROOT / "models" / "whisper"))
        segments, _ = recognizer.transcribe(str(run / "reference.wav"), language="en", beam_size=5)
        reference_text = " ".join(s.text.strip() for s in segments).strip()
        del recognizer
    if not reference_text:
        raise RuntimeError("No reference transcript obtained. Supply --ref-text.")
    print("Reference transcript:", reference_text, flush=True)
    (run / "reference.txt").write_text(reference_text, encoding="utf-8")
    config = dict(sentences=sentences, reference_text=reference_text,
                  models=args.models, seed=args.seed, start=args.start, duration=args.duration)
    (run / "settings.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    statuses = {}
    for model in args.models:
        print(f"Running {MODELS[model]} (weights download on first use)...", flush=True)
        # Separate processes release GPU memory completely between models.
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                 "--worker", str(run), "--model", model, "--device", args.device])
        statuses[model] = result.returncode
        if result.returncode:
            print(f"{model} failed. If CUDA ran out of memory, retry --models {model} --device cpu.", flush=True)
    page = ['<!doctype html><meta charset="utf-8"><title>Voice comparison</title>',
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}td,th{padding:12px;text-align:left}audio{width:300px}</style>',
            '<h1>Voice comparison</h1><p>Compare accent, voice similarity, clarity, and naturalness. Model size alone does not determine the best result.</p>',
            '<h2>Reference</h2><audio controls src="reference.wav"></audio>',
            '<p>' + html.escape(reference_text) + '</p>',
            '<table><tr><th>Sentence</th>' + ''.join('<th>' + html.escape(m) + '</th>' for m in args.models) + '</tr>']
    for i, sentence in enumerate(sentences, 1):
        cells = []
        for model in args.models:
            relative = f"{model}/{i:02d}.wav"
            cells.append(f'<td><audio controls preload="none" src="{relative}"></audio></td>'
                         if (run / relative).exists() else '<td>Generation failed</td>')
        page.append('<tr><td>' + html.escape(sentence) + '</td>' + ''.join(cells) + '</tr>')
    page.append('</table>')
    (run / "compare.html").write_text("\n".join(page), encoding="utf-8")
    (run / "status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    print(f"Open this listening page: {run / 'compare.html'}", flush=True)
    return int(any(statuses.values()))


if __name__ == "__main__":
    raise SystemExit(main())

