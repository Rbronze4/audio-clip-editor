"""Isolated inference process. Only the official htdemucs model is loaded."""
import argparse
import json
from pathlib import Path
import sys
import time
import wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--status', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    status = Path(args.status)

    def report(progress, message, **extra):
        temporary = status.with_suffix('.new')
        temporary.write_text(json.dumps(dict(progress=progress, message=message, **extra),
                                        ensure_ascii=False), encoding='utf-8')
        for attempt in range(10):
            try:
                temporary.replace(status)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.02)

    try:
        report(-1, 'Demucs・PyTorchを起動中…')
        try:
            import numpy as np
            import soundfile as sf
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
        except ImportError as error:
            raise RuntimeError('BGM除去用パッケージがありません。setup_bgm.batを実行してください。') from error
        torch.hub.set_dir(str(Path(args.models).resolve()))
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDAを利用できません。NVIDIAドライバとGPU用PyTorchを確認するか、処理方式をCPUにしてください。')
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        label = ('CUDA: ' + torch.cuda.get_device_name(0)) if device == 'cuda' else 'CPU'
        report(-1, f'モデルを準備中… 初回はダウンロードします / {label}', device=label)
        try:
            model = get_model('htdemucs')
        except Exception as error:
            raise RuntimeError(f'分離モデルを読み込めません。初回はインターネット接続が必要です。\n{error}') from error
        model.eval()
        if args.prepare:
            report(100, f'BGM除去の準備完了 / {label}', device=label)
            return
        model.to(device)
        rate = model.samplerate
        with wave.open(args.input, 'rb') as wav, sf.SoundFile(
                args.output, 'w', samplerate=rate, channels=model.audio_channels,
                format='WAV', subtype='FLOAT') as output:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (rate, model.audio_channels, 2):
                raise RuntimeError('分離用PCMの形式が一致しません。')
            total = wav.getnframes()
            # Bound host RAM even for hour-long files. One second of context on
            # each side is inferred too, then discarded to keep exact alignment.
            stride, context = rate * 30, rate
            for start in range(0, total, stride):
                end = min(total, start + stride)
                left, right = max(0, start - context), min(total, end + context)
                wav.setpos(left)
                samples = np.frombuffer(wav.readframes(right - left), dtype='<i2').reshape(-1, model.audio_channels)
                mix = torch.from_numpy(samples.astype(np.float32).T.copy()) / 32768
                mean, std = mix.mean(), mix.std()
                if std.item() < 1e-8:
                    vocals = torch.zeros_like(mix)
                else:
                    with torch.inference_mode():
                        result = apply_model(model, ((mix - mean) / std)[None], device=device,
                                             shifts=0, split=True, overlap=.25,
                                             progress=False, num_workers=0, segment=7)[0]
                        vocals = result[model.sources.index('vocals')].cpu() * std + mean
                samples = vocals[:, start-left:end-left].T.contiguous().numpy()
                if not np.isfinite(samples).all():
                    raise RuntimeError('モデルが不正な音声値を返しました。')
                output.write(samples)
                report(int(end * 100 / total), f'BGM除去中… {end / rate:.1f} / {total / rate:.1f}秒 / {label}', device=label)
        report(100, f'音源分離完了 / {label}', device=label)
    except Exception as error:
        message = str(error)
        if 'cuda' in message.lower() or 'out of memory' in message.lower():
            message += '\nGPUメモリ不足・CUDAエラーの場合は、範囲を短くするかCPUで再実行してください。'
        report(-1, message, error=True)
        print(message, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
