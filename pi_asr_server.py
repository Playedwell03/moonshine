#!/usr/bin/env python3
import argparse
import os
import socket
import struct
import tempfile
import time
import wave
import time as _time


MAGIC = b"MSH1"


def parse_args():
    parser = argparse.ArgumentParser(description="Receive mic audio and run Moonshine ASR.")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5050, help="TCP port (default: 5050).")
    parser.add_argument("--language", default="ko", help="Language code (default: ko).")
    parser.add_argument("--chunk-seconds", type=float, default=4.0, help="Chunk length in seconds.")
    parser.add_argument("--print-latency", action="store_true", help="Print per-chunk latency.")
    parser.add_argument("--model-path", default=None, help="Override model path.")
    parser.add_argument("--model-arch", default=None, help="Override model arch (e.g., TINY, BASE).")
    return parser.parse_args()


def recv_exact(conn, size):
    buf = b""
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def write_wav(path, pcm_bytes, sample_rate, channels):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _normalize_text(result):
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    if hasattr(result, "lines"):
        try:
            return " ".join(
                line.text for line in result.lines if getattr(line, "text", None)
            ).strip()
        except Exception:
            pass
    return str(result)


def _parse_model_arch(mv, value):
    if value is None:
        return None
    try:
        return getattr(mv.moonshine_api.ModelArch, value.upper())
    except Exception:
        return None


def transcribe_wav(path, language, model_path_override=None, model_arch_override=None):
    try:
        from moonshine_voice import get_model_for_language
        import moonshine_voice as mv
        from moonshine_voice.utils import load_wav_file
    except Exception as exc:
        raise RuntimeError("moonshine_voice import failed") from exc

    if model_path_override:
        model_path = model_path_override
        model_arch = model_arch_override or mv.moonshine_api.ModelArch.TINY
    else:
        model_path, model_arch = get_model_for_language(language)
        if model_arch_override is not None:
            model_arch = model_arch_override
    transcriber = mv.transcriber.Transcriber(model_path=model_path, model_arch=model_arch)
    audio_data, sample_rate = load_wav_file(path)
    result = transcriber.transcribe_without_streaming(audio_data, sample_rate=sample_rate)
    return _normalize_text(result)


def main():
    args = parse_args()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)

    print(f"Listening on {args.bind}:{args.port} ...", flush=True)
    conn, addr = server.accept()
    print(f"Client connected: {addr}", flush=True)

    header = recv_exact(conn, 4 + 4 + 2 + 2)
    magic, sample_rate, channels, bytes_per_sample = struct.unpack("<4sIHH", header)
    if magic != MAGIC:
        raise RuntimeError("Invalid stream header")
    if bytes_per_sample != 2:
        raise RuntimeError("Only int16 PCM is supported")

    bytes_per_second = int(sample_rate * channels * bytes_per_sample)
    chunk_bytes = int(bytes_per_second * args.chunk_seconds)
    buffer = bytearray()

    try:
        while True:
            length_bytes = recv_exact(conn, 4)
            (length,) = struct.unpack("<I", length_bytes)
            payload = recv_exact(conn, length)
            buffer.extend(payload)

            while len(buffer) >= chunk_bytes:
                chunk = bytes(buffer[:chunk_bytes])
                del buffer[:chunk_bytes]
                chunk_ready_ts = _time.monotonic()

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name

                write_wav(wav_path, chunk, sample_rate, channels)
                try:
                    import moonshine_voice as mv
                    model_arch = _parse_model_arch(mv, args.model_arch)
                    text = transcribe_wav(
                        wav_path,
                        args.language,
                        model_path_override=args.model_path,
                        model_arch_override=model_arch,
                    )
                    if text:
                        if args.print_latency:
                            latency_ms = (_time.monotonic() - chunk_ready_ts) * 1000.0
                            print(f"{text}  [latency_ms={latency_ms:.0f}]", flush=True)
                        else:
                            print(text, flush=True)
                except Exception as exc:
                    print(f"[transcribe error] {exc}", flush=True)
                finally:
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
    except KeyboardInterrupt:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
