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