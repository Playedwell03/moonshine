# Moonshine Realtime

Moonshine 기반 실시간 음성 인식 실행 가이드입니다.

## Setup (공통)

```bash
cd ~/moonshine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

라즈베리파이에서 USB 마이크를 쓸 때(최초 1회):

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev alsa-utils
```

입력 디바이스 인덱스 확인:

```bash
source .venv/bin/activate
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## 1) 기본 실시간 버전 (`moonshine_realtime.py`)

원문 인식 결과를 실시간으로 출력합니다.

```bash
cd ~/moonshine
source .venv/bin/activate
python3 moonshine_realtime.py --language ko --device <DEVICE_IDX> --samplerate 48000 --channels 1 --blocksize 4096 --update-interval 1.0 --final-only --merge-window 0.8 --min-chars 2
```

---

## 2) 후처리 매핑 버전 (`moonshine_realtime_mapped.py`)

오인식 표현을 지정한 명령어로 보정합니다.

```bash
cd ~/moonshine
source .venv/bin/activate
python3 moonshine_realtime_mapped.py --language ko --device <DEVICE_IDX> --samplerate 48000 --channels 1 --blocksize 4096 --update-interval 1.0 --final-only --merge-window 0.8 --min-chars 2
```

커스텀 매핑 추가 예시:

```bash
cd ~/moonshine
source .venv/bin/activate
python3 moonshine_realtime_mapped.py --language ko --device <DEVICE_IDX> --samplerate 48000 --final-only --command-map "창문 열어줘=창문열어줘,창문 여로줘,상문 봐,창문 봐"
```

---

## 3) 강제 4-클래스 분류 버전 (`moonshine_realtime_forced.py`)

인식 결과를 아래 4개 중 하나로 강제 분류해서 출력합니다.

1. `창문 열어줘`
2. `창문 닫아줘`
3. `창문 투명하게 해줘`
4. `창문 불투명하게 해줘`

```bash
cd ~/moonshine
source .venv/bin/activate
python3 moonshine_realtime_forced.py --language ko --device <DEVICE_IDX> --samplerate 48000 --channels 1 --blocksize 4096 --update-interval 1.0 --final-only --merge-window 0.8 --min-chars 2
```

---

## 4) 맥 마이크 -> 라즈베리파이 네트워크 버전

### 4-1) Raspberry Pi (ASR 서버)

```bash
cd ~/moonshine
source .venv/bin/activate
python3 pi_asr_server.py --language ko --port 5050
```

### 4-2) Mac (마이크 송신)

```bash
cd /Users/playedwell/moons
source .venv/bin/activate
python3 mic_streamer.py --host <PI_IP> --port 5050
```

---

<<<<<<< Updated upstream
=======
## 5) 파일 배포 (Mac -> Raspberry Pi)

맥에서 수정한 스크립트를 Pi로 복사:

```bash
cd /Users/playedwell/moons
scp moonshine_realtime_mapped.py wind@<PI_IP>:~/moonshine/
scp moonshine_realtime_forced.py wind@<PI_IP>:~/moonshine/
```

`detect1` 같은 호스트명이 안 되면 `detect1.local` 또는 `<PI_IP>`를 사용하세요.

---

>>>>>>> Stashed changes
## Notes

- `GPU device discovery failed` 경고는 Pi에서 자주 보이는 ONNXRuntime 경고로, 보통 무시해도 됩니다.
- `PaErrorCode -9997 (Invalid sample rate)`가 나면 `--samplerate`를 마이크 기본값(대개 `48000`)으로 맞추세요.
- `MicTranscriber: input overflow`가 잦으면 `--blocksize 4096`, `--update-interval 1.0`처럼 처리 부담을 낮추세요.
- 종료는 `Ctrl+C`.
