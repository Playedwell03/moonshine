
환경 : 라즈베리파이 로컬 + 맥북

# 🌬️ RX-9M Sensor Monitoring Project

라즈베리파이 4와 RX-9M 센서를 활용하여 실시간으로 실내 공기 질(CO2, PM2.5)을 모니터링하는 시스템입니다. 맥북에서 SSH로 접속하여 터미널 환경에서 데이터를 확인하는 과정을 포함합니다.

## 1. 하드웨어 구성 (Hardware Setup)

### 🔌 핀 연결 안내
RX-9M 센서의 뒷면 레이블(**V, G, E, T**)을 확인하여 빵판과 점퍼 와이어를 이용해 아래와 같이 연결합니다.

| 센서 (RX-9M) | 라즈베리파이 4 (GPIO) | 핀 번호 | 기능 |
| :--- | :--- | :--- | :--- |
| **V** (VCC) | 5V Power | Pin 2 | 전원 공급 (발열 발생은 정상) |
| **G** (GND) | Ground | Pin 6 | 접지 |
| **E** (Enable/RX) | GPIO 14 (TXD) | Pin 8 | 데이터 수신 (파이 TX -> 센서 RX) |
| **T** (TX) | GPIO 15 (RXD) | Pin 10 | 데이터 송신 (센서 TX -> 파이 RX) |

> **⚠️ 중요**: 시리얼 통신은 송신(TX)과 수신(RX)을 반드시 교차(Cross) 연결해야 합니다.

---

## 2. 라즈베리파이 설정 (OS Configuration)

시리얼 통신 기능을 활성화하기 위해 터미널(SSH)에서 다음 설정을 수행합니다.

### 시리얼 포트 활성화
1. **설정 도구 실행**
   ```bash
   sudo raspi-config
   
2. Interface Options -> Serial Port 선택
    
3. Login shell accessible over serial? -> `No` (시스템 콘솔 점유 해제)
    
4. Serial port hardware enabled? -> `Yes` (하드웨어 포트 활성화)
    
5. 재부팅 (필수)
    
    Bash
    
    ```
    sudo reboot
    ```
    

---

## 3. 모니터링 스크립트 (Python Code)

### 라이브러리 설치

Bash

```
pip install pyserial
```

### 코드 작성 (`read_sensor.py`)

`nano read_sensor.py` 명령어로 파일을 생성한 후 아래 코드를 붙여넣습니다.

Python

```
import serial
import time

# 라즈베리파이 4의 기본 하드웨어 시리얼 포트 설정
# 연결 상태에 따라 /dev/ttyS0 또는 /dev/ttyAMA0를 사용합니다.
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)

print("--- RX-9M Real-time Monitoring Started ---")

try:
    while True:
        # 패킷 시작 바이트(0x42, 0x4D) 감지
        if ser.read() == b'\x42':
            if ser.read() == b'\x4d':
                # 헤더 이후의 유효 데이터 10바이트를 읽음
                data = ser.read(10)
                
                if len(data) >= 10:
                    # 16진수 데이터를 10진수 수치로 변환 (High * 256 + Low)
                    co2 = data[2] * 256 + data[3]
                    pm25 = data[4] * 256 + data[5]
                    
                    # 결과 출력
                    print(f"[{time.strftime('%H:%M:%S')}] CO2: {co2:4} ppm | PM2.5: {pm25:3} ug/m3")
        
        time.sleep(1) # 1초 간격 모니터링

except KeyboardInterrupt:
    print("\n--- Monitoring Stopped by User ---")
finally:
    ser.close()
```

---

## 4. 실행 및 모니터링 (Usage)

1. **맥북 터미널에서 라즈베리파이 접속**
    
    Bash
    
    ```
    ssh wind@detect1.local
    ```
    
2. **스크립트 실행**
    
    Bash
    
    ```
    python3 read_sensor.py
    ```
    

---

## 5. 문제 해결 (Troubleshooting)

- 데이터가 나오지 않을 때: `ls -l /dev/serial*`을 입력하여 `serial0`이 어떤 포트(`ttyS0` 또는 `ttyAMA0`)에 연결되어 있는지 확인 후 코드의 `ser = serial.Serial(...)` 부분을 수정하세요.
    
- 값이 너무 높게 나올 때: 센서 특성상 내부 챔버 예열이 필요합니다. 전원 연결 후 약 5~10분 정도 기다리면 수치가 정상 범위로 안정화됩니다.
    
- 외계어가 출력될 때: 센서는 바이너리 데이터를 전송하므로 `ser.readline()`을 쓰면 깨집니다. 반드시 위 코드처럼 `ser.read()`를 사용하여 바이트 단위로 처리해야 합니다.