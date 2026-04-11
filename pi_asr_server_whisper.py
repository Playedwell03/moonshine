#!/usr/bin/env python3
import argparse
import os
import socket
import struct
import tempfile
import wave

from faster_whisper import WhisperModel


MAGIC = b"MSH1"


def parse_args():
    parser = argparse.ArgumentParser(description="Receive mic audio and run Whisper ASR.")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5050, help="TCP port (default: 5050).")
    parser.add_argument("--language", default="ko", help="Language code (default: ko).")
    parser.add_argument("--chunk-seconds", type=float, default=4.0, help="Chunk length in seconds.")
    parser.add_argument("--model", default="small", help="Whisper model size (default: small).")
    parser.add_argument("--compute-type", default="int8", help="Compute type (default: int8).")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu).")
    parser.add_argument("--vad", action="store_true", help="Enable VAD filtering.")
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


def main():
    args = parse_args()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

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

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name

                write_wav(wav_path, chunk, sample_rate, channels)
                try:
                    segments, _info = model.transcribe(
                        wav_path, language=args.language, vad_filter=args.vad
                    )
                    text = "".join(seg.text for seg in segments).strip()
                    if text:
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
