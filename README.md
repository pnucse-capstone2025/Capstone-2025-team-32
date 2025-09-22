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
| 정연수     | example01@pusan.ac.kr   | 프로젝트 총괄, 드론 하드웨어 구성 , VPN 네트워크 구성|
| 김현태     | tee9665@pusan.ac.kr   | Jetson Nano 환경 세팅, AI 모델 훈련및 최적화 , VPN 네트워크 구성|


---

## 3. 구성도
### 동작 흐름도
1. **Camera (Intel RealSense D435i)** → 타겟 인식  
2. **Jetson Nano (YOLOv5/TensorRT)** → 객체 탐지 및 판단  
3. **Pixhawk (Flight Controller)** → 모터 제어, 비행 모드 전환  
4. **Drone (프레임 + 모터 + 배터리)** → 실제 이동 및 제어  
5. **Ground Station (Mission Planner, ffplay)** → 실시간 모니터링 및 제어  

