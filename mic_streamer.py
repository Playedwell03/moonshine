#!/usr/bin/env python3
import argparse
import queue
import socket
import struct
import sys
import time

import numpy as np
import sounddevice as sd


MAGIC = b"MSH1"


def parse_args():
    parser = argparse.ArgumentParser(description="Stream Mac mic audio to a Raspberry Pi over TCP.")
    parser.add_argument("--host", required=True, help="Raspberry Pi IP/hostname.")
    parser.add_argument("--port", type=int, default=5050, help="TCP port (default: 5050).")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (default: 16000).")
    parser.add_argument("--channels", type=int, default=1, help="Channels (default: 1).")
    parser.add_argument("--block-size", type=int, default=1024, help="Frames per block (default: 1024).")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_queue = queue.Queue(maxsize=100)

    def callback(indata, frames, time_info, status):
        if status:
            pass
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    print(f"Connecting to {args.host}:{args.port} ...", flush=True)
    sock = socket.create_connection((args.host, args.port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    header = struct.pack("<4sIHH", MAGIC, args.sample_rate, args.channels, 2)
    sock.sendall(header)
    print("Streaming... Press Ctrl+C to stop.", flush=True)

    with sd.InputStream(
        samplerate=args.sample_rate,
        channels=args.channels,
        dtype="int16",
        blocksize=args.block_size,
        callback=callback,
    ):
        try:
            while True:
                block = audio_queue.get()
                payload = block.tobytes()
                sock.sendall(struct.pack("<I", len(payload)))
                sock.sendall(payload)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
