# Moonshine Realtime (Mac mic)

This is a tiny CLI that streams your Mac microphone into the Moonshine ASR
model and prints live text to the terminal.

## Setup

1. Create a venv (optional) and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run realtime transcription:

```bash
python3 moonshine_realtime.py --language ko
```

## Mac mic -> Raspberry Pi (network)

Use your Mac as the microphone and run ASR on the Raspberry Pi.

1. On Raspberry Pi (receiver + ASR):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 pi_asr_server.py --language ko --port 5050
```

2. On Mac (mic sender):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 mic_streamer.py --host <PI_IP> --port 5050
```

3. Speak into the Mac mic; transcription appears on the Pi.

## Notes

- macOS will prompt for microphone permission the first time you run it.
- Stop with Ctrl+C.
- The script uses `get_model_for_language("ko")`, which resolves to the
  `moonshine-tiny-ko` model under the hood.
- If you later attach a mic to the Pi, you can run `moonshine_realtime.py`
  directly on the Pi instead of using the network sender/receiver.
