# 자폭드론 
ASTRO (정연수(202255671) , 김현태(202055529))

# 재난자 구조 및 자폭 드론 프로젝트

## 1. 프로젝트 소개
- **프로젝트명:** 재난자 구조 및 자폭 드론 (Rescue & Self-Destruct Drone)  
- **목적:**  
  재난 현장에서 신속히 생존자를 탐지하고 구조 상황을 지원하며, 위험한 상황에서는 자폭 기능을 통해 2차 피해를 방지하는 것을 목표로 한다.  
- **개요:**  
  본 프로젝트는 **Intel RealSense 카메라 + Jetson Nano + Pixhawk** 기반으로 동작한다.  
  - **YOLOv5 모델**을 통한 실시간 객체 인식  
  - **Pixhawk**를 통한 드론 모터 제어  
  - **Mission Planner**를 통한 자동 비행 및 Guided 모드 전환  
  - **UDP 스트리밍**을 통한 실시간 영상 전송 및 원격 제어 지원  

---

## 2. 팀 소개
| 이름       | 이메일                  | 역할                          |
|------------|-------------------------|-------------------------------|
| 정연수     | aden1213@pusan.ac.kr   | 프로젝트 총괄, 드론 하드웨어 구성 , VPN 네트워크 구성|
| 김현태     | tee9665@pusan.ac.kr   | Jetson Nano 환경 세팅, AI 모델 훈련및 최적화 , VPN 네트워크 구성|

1. **Camera (Intel RealSense D435i)** → 타겟 인식  
2. **Jetson Nano (YOLOv5/TensorRT)** → 객체 탐지 및 판단  
3. **Pixhawk (Flight Controller)** → 모터 제어, 비행 모드 전환  
4. **Drone (프레임 + 모터 + 배터리)** → 실제 이동 및 제어  
5. **Ground Station (Mission Planner, ffplay)** → 실시간 모니터링 및 제어  


---

## 3. 구성도
### 동작 흐름도
<img width="1024" height="498" alt="image" src="https://github.com/user-attachments/assets/61c8828c-280b-4e1a-9ade-bca0df6b73e4" />


## 4. 네트워크 
본 프로젝트는 **재난 현장 원격 제어 및 모니터링**을 위해 네트워크 기반 구조를 갖춘다.  
- **UDP 스트리밍**: 지연(latency)을 최소화하기 위해 TCP 대신 UDP 프로토콜을 사용한다.  
- **Ground Station ↔ Drone**: 상호 간의 명령 및 영상 데이터 교환은 **양방향 통신**을 기반으로 한다.  
- **실시간 영상 전송**: Jetson Nano에서 YOLO 객체 인식 결과를 인코딩(H.264)하여 Ground Station으로 송출한다.  
- **원격 제어**: Ground Station에서 Mission Planner를 통해 Pixhawk로 명령 전송.  

<img width="819" height="546" alt="image" src="https://github.com/user-attachments/assets/0788692c-4c42-4731-892f-8cc91da23c00" />




## 5. YOLO 와 Tensorrt 최적화

- **YOLOv5/TensorRT 엔진**을 활용하여 실시간으로 재난 현장의 타겟(사람, 물체 등)을 탐지한다.  
- **TensorRT 최적화**: Jetson Nano 환경에서 FP16/INT8 모델을 활용해 지연을 최소화한다.  
- **탐지 결과 활용**:  
  - 탐지된 객체의 **Bounding Box 중심 좌표**를 계산  
  - 해당 좌표값을 Pixhawk로 전달하여 드론의 위치 제어에 반영  
- **추가 기능**:  
  - YOLO Confidence 값 기반으로 **임계값(Threshold 70%)**을 설정  
  - 일정 시간(예: 2초 이상) 조건 충족 시 **Guided 모드 전환** 및 **모터 제어** 수행  

<img width="940" height="590" alt="image" src="https://github.com/user-attachments/assets/200ff9df-b63c-493f-aed6-d30ee786ed2a" />
<img width="942" height="332" alt="image" src="https://github.com/user-attachments/assets/21eef1bb-19dd-460f-91e8-7f7275a0678e" />

- **YOLO 돌린 화면**:
<img width="881" height="468" alt="image" src="https://github.com/user-attachments/assets/232d2e87-bb40-4a05-b614-5e6ba4c5e8f8" />





