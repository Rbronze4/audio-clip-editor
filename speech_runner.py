"""48 kHz speech enhancement in an isolated, cancelable process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import urllib.request

MODEL_REVISION = 'eff8c97925c8bec812af707814b3e5d777fd4503'
MODEL_SHA256 = '03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6'
MODEL_URL = ('https://huggingface.co/alibabasglab/MossFormer2_SE_48K/resolve/'
             + MODEL_REVISION + '/last_best_checkpoint.pt')


def checksum(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ensure_model(directory, report):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'mossformer2_se_48k.pt'
    if target.is_file() and checksum(target) == MODEL_SHA256:
        return target
    report(-1, '会話向けモデルを取得中…（初回約222MB）')
    fd, temporary = tempfile.mkstemp(suffix='.part', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            while block := response.read(1024 * 1024):
                output.write(block)
        if checksum(temporary) != MODEL_SHA256:
            raise RuntimeError('会話向けモデルの検証に失敗しました。再度セットアップしてください。')
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def enhance_block(model, device, samples, config):
    import numpy as np
    import torch
    from clearvoice.utils.decode import decode_one_audio_mossformer2_se_48k
    length = len(samples)
    # STFT/fbank/delta context requires padding for very short selections.
    padded = np.pad(samples, ((0, max(0, 24000-length)), (0, 0)))
    output = np.zeros_like(padded)
    with torch.inference_mode():
        for channel in range(samples.shape[1]):
            signal = padded[:, channel]
            if np.max(np.abs(signal)) < 1e-7:
                continue
            result = decode_one_audio_mossformer2_se_48k(model, device, signal[None], config)
            output[:, channel] = result[:len(padded)]
    output = output[:length]
    if not np.isfinite(output).all():
        raise RuntimeError('会話向けモデルが不正な音声値を返しました。')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--models', required=True)
    parser.add_argument('--status', required=True)
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
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.02)

    try:
        report(-1, '会話向けの音声処理を準備中…')
        directory = Path(args.models).resolve() / 'speech'
        os.environ['NUMBA_CACHE_DIR'] = str(directory / 'numba-cache')
        try:
            import soundfile as sf
            import torch
            from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
        except ImportError as error:
            raise RuntimeError('会話向けパッケージがありません。setup_speech.batを実行してください。') from error
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDAを利用できません。CPUを選ぶかNVIDIAドライバを確認してください。')
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        label = 'CUDA: ' + torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
        path = ensure_model(directory, report)
        config = SimpleNamespace(sampling_rate=48000, one_time_decode_length=20, decode_window=4,
                                 win_type='hamming', win_len=1920, win_inc=384, fft_len=1920, num_mels=60)
        model = MossFormer2_SE_48K(config).model
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        if args.prepare:
            report(100, f'会話向けBGM低減の準備完了 / {label}')
            return
        with sf.SoundFile(args.input) as audio, sf.SoundFile(
                args.output, 'w', samplerate=48000, channels=audio.channels,
                subtype='FLOAT', format='WAV') as output:
            if audio.samplerate != 48000 or audio.channels not in (1, 2):
                raise RuntimeError('会話向けの入力形式が一致しません。')
            stride, context = 48000 * 10, 48000
            total = len(audio)
            for start in range(0, total, stride):
                end = min(total, start + stride)
                left, right = max(0, start-context), min(total, end+context)
                audio.seek(left)
                samples = audio.read(right-left, dtype='float32', always_2d=True)
                result = enhance_block(model, device, samples, config)
                output.write(result[start-left:end-left])
                report(int(end*100/total), f'会話向けBGM低減… {end/48000:.1f} / {total/48000:.1f}秒 / {label}')
        report(100, '会話向け処理完了 — A/Bで声と背景音を確認してください')
    except Exception as error:
        message = str(error)
        if 'cuda' in message.lower() or 'out of memory' in message.lower():
            message += '\nCPUを選ぶか、対象範囲を短くして再実行してください。'
        report(-1, message, error=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
