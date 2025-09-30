#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hold_descend5_webcam.py (Hybrid)
- 입력: /dev/video0 (USB 웹캠, GStreamer v4l2src + appsink)
- 추론: YOLOv5 TensorRT (엔진/이름/임계값 web_cam_1080.py와 동일 가정)
- 송출: NVENC → H.264 → RTP/UDP → udpsink (web_cam_1080.py와 동일 체인)
- 트리거: 최근 2초 창에서 70% 이상 프레임에 검출시 동작
    * ACTION_MODE="hold"    : GUIDED 전환 후 제자리 정지
    * ACTION_MODE="descend5": GUIDED 전환 후 홈 기준 상대고도 TARGET_REL_ALT_M까지 하강
- 차이점(하이브리드):
    1) GLOBAL_POSITION_INT 텔레메트리 5Hz 요청 추가 (안정적 rel_alt 확보)
    2) POSITION 전용 type_mask는 A 버전(표준적/안전) 유지
"""

import os, time, json, threading, cv2, numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import tensorrt as trt
import pycuda.driver as cuda

from utils.plots import Annotator, colors

# ===================== 선택 동작 =====================
# "hold" 또는 "descend5"
ACTION_MODE = "descend5"

# ===================== 사용자 설정 (web_cam_1080.py와 동일) =====================
GROUND_IP    = "10.0.0.4"        # 지상 PC IP
PORT         = 5000
ENGINE_PATH  = "runs/train/1490_images_1280_b164/weights/best_int8.engine"
CONF_THR     = 0.30
IOU_THR      = 0.45
FPS          = 15

# 해상도/디바이스
COLOR_WIDTH, COLOR_HEIGHT = 1280, 720
DEVICE_PATH  = "/dev/video0"
NAMES = ["Basket"]     # 단일 클래스 가정

# 트리거 파라미터
WINDOW_SEC      = 2.0          # 투표 윈도우(초)
HIT_CONF_MIN    = 0.50         # YOLO conf 최소
TRIGGER_RATIO   = 0.70         # 70%
SUSTAIN_SEC     = 0.4          # 임계 유지 시간(초)
COOLDOWN_SEC    = 65.0         # 재발동 쿨다운(초)
STATUS_JSON     = "/tmp/yolo_kofn.json"

# MAVLink 연결(필요 시 수정)
SERIAL_PORT       = "/dev/ttyTHS1"
SERIAL_BAUD       = 57600
HOLD_DURATION_S   = 1.0        # hold 모드에서 0속도 명령 지속
HOLD_RATE_HZ      = 5
TARGET_REL_ALT_M  = 5.0        # 목표 상대고도(m)
DESCEND_SPEED_MPS = 0.6        # (참고값) 하강 속도, 실제론 position setpoint 사용
DESCEND_TIMEOUT_S = 60
REACH_EPS_M       = 0.15

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# ===================== TensorRT 유틸 =====================
def load_engine(path: str):
    print(f"🚀 Loading TensorRT engine from: {path}")
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("❌ TensorRT 엔진 로드 실패")
    print("✅ TensorRT Engine Loaded.")
    return engine

def get_engine_input_shape(engine):
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            shp = list(engine.get_binding_shape(i))
            if shp[0] < 0: shp[0] = 1
            if len(shp) == 4:
                n, c, h, w = map(int, shp)
                return n, c, h, w, i
    raise RuntimeError("No input binding found")

def allocate_buffers(engine):
    host_in = device_in = None
    host_out, device_out, bindings = [], [], []
    stream = cuda.Stream()
    for i in range(engine.num_bindings):
        shp = list(engine.get_binding_shape(i))
        if shp[0] < 0: shp[0] = 1
        size  = trt.volume(shp)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        hmem  = cuda.pagelocked_empty(size, dtype)
        dmem  = cuda.mem_alloc(hmem.nbytes)
        bindings.append(int(dmem))
        if engine.binding_is_input(i):
            host_in, device_in = hmem, dmem
        else:
            host_out.append(hmem)
            device_out.append(dmem)
    return (host_in, device_in), (host_out, device_out, bindings, stream)

# ===================== 후처리 =====================
def xywh2xyxy(x: np.ndarray):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms_numpy(pred, conf_thres=0.25, iou_thres=0.45):
    out = [np.zeros((0, 6), dtype=np.float32)]
    x = pred[0]
    if x.size == 0: return out
    x = x[x[:, 4] > conf_thres]
    if not x.size: return out
    boxes  = xywh2xyxy(x[:, :4])
    scores = x[:, 4]
    cls    = np.zeros_like(scores)
    b4nms  = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
    idxs   = cv2.dnn.NMSBoxes(b4nms, scores.tolist(), conf_thres, iou_thres)
    if hasattr(idxs, "flatten"):
        idxs = idxs.flatten()
        final = np.concatenate((boxes[idxs], scores[idxs, None], cls[idxs, None]), 1)
        out[0] = final.astype(np.float32)
    return out

def scale_coords(img1_shape, coords, img0_shape):
    gw = img1_shape[1] / img0_shape[1]
    gh = img1_shape[0] / img0_shape[0]
    coords[:, [0, 2]] /= gw
    coords[:, [1, 3]] /= gh
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    return coords

# ===================== YOLO 파이프라인 =====================
class Pipeline:
    def __init__(self, gpu_lock, context, executor, engine, net_h, net_w, in_binding_idx):
        self.gpu_lock = gpu_lock
        self.context  = context
        self.gpu_exec = executor
        self.net_h = net_h
        self.net_w = net_w
        self.in_binding_idx = in_binding_idx
        self.img_rs      = np.empty((self.net_h, self.net_w, 3), dtype=np.uint8)
        self.img_rgb     = np.empty((self.net_h, self.net_w, 3), dtype=np.uint8)
        self.img_rgb_f32 = np.empty((self.net_h, self.net_w, 3), dtype=np.float32)
        self._init_future = executor.submit(self._init_buffers, engine)
        self._run_future = None

    def _init_buffers(self, engine):
        (self.h_in, self.d_in), (self.h_outs, self.d_outs, self.bindings, self.stream) = allocate_buffers(engine)

    def ensure_ready(self):
        if self._init_future is not None:
            self._init_future.result()
            self._init_future = None

    def preprocess(self, img0):
        cv2.resize(img0, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR, dst=self.img_rs)
        cv2.cvtColor(self.img_rs, cv2.COLOR_BGR2RGB, dst=self.img_rgb)
        np.divide(self.img_rgb, 255.0, out=self.img_rgb_f32, dtype=np.float32)
        self.h_in[:] = self.img_rgb_f32.transpose(2, 0, 1).ravel()

    def gpu(self):
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for h, d in zip(self.h_outs, self.d_outs):
            cuda.memcpy_dtoh_async(h, d, self.stream)
        self.stream.synchronize()

    def postprocess(self, img0):
        outs = []
        for h in self.h_outs:
            arr = np.array(h, copy=False).reshape(-1, 6)
            outs.append(arr)
        if not outs:
            return img0, []
        preds_raw = np.concatenate(outs, 0)[None, ...]
        preds = nms_numpy(preds_raw, CONF_THR, IOU_THR)[0]
        ann = Annotator(img0, line_width=2, pil=True)
        dets = []
        if preds.size:
            preds[:, :4] = scale_coords((self.net_h, self.net_w), preds[:, :4], img0.shape)
            for *xyxy, conf, cls in preds:
                x1, y1, x2, y2 = map(int, xyxy)
                dets.append((x1, y1, x2, y2, float(conf)))
                label = f"{NAMES[int(cls)]} {conf:.2f}"
                ann.box_label((x1, y1, x2, y2), label, colors(int(cls), True))
        return ann.result(), dets

    def run(self, img0):
        self.ensure_ready()
        self.preprocess(img0)
        self.gpu_lock.acquire()
        try:
            self.gpu_exec.submit(Pipeline.gpu, self).result()
        finally:
            self.gpu_lock.release()
        return self.postprocess(img0)

    def submit(self, executor, img0):
        self._run_future = executor.submit(Pipeline.run, self, img0)

    def wait(self):
        if self._run_future is None:
            return None
        res = self._run_future.result()
        self._run_future = None
        return res

# ===================== MAVLink 동작 =====================

# --- 추가: 전역 공유 연결 상태 ---
_MAV = None
_MAVUTIL = None
_MAV_LOCK = threading.Lock()

def _connect_mav():
    """
    전역에서 한 번만 연결을 맺고, 이후에는 같은 객체를 반환.
    기존 호출부를 전혀 바꾸지 않기 위해 반환 형태(m, mavutil)는 그대로 유지.
    각 스레드의 finally에서 m.close()를 호출해도 실제로 닫히지 않도록
    close()를 no-op으로 패치한다.
    """
    global _MAV, _MAVUTIL
    with _MAV_LOCK:
        if _MAV is None:
            from pymavlink import mavutil as _mu
            _MAVUTIL = _mu
            _MAV = _mu.mavlink_connection(SERIAL_PORT, baud=SERIAL_BAUD)
            _MAV.wait_heartbeat(timeout=5)

            # 공유 연결이 실수로 닫히지 않도록 no-op close
            def _noop_close():
                pass
            _MAV.close = _noop_close
    return _MAV, _MAVUTIL

def _set_mode_guided(m, mavutil):
    print("[MAV] Set GUIDED ...")
    try:
        mavutil.set_mode(m, "GUIDED")
    except Exception:
        GUIDED = 4
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            GUIDED, 0,0,0,0,0
        )
    # 확인(최대 3초)
    t0 = time.time()
    ok = False
    while time.time() - t0 < 3.0:
        hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
        if hb and hasattr(hb, "custom_mode") and hb.custom_mode == 4:
            ok = True; break
    print(f"[MAV] GUIDED: {'OK' if ok else 'NOT CONFIRMED'}")
    return ok

def _send_local_vel(m, mavutil, vx, vy, vz, yaw_rate=0.0):
    type_mask = (1<<0)|(1<<1)|(1<<2)|(1<<6)|(1<<7)|(1<<8)|(1<<9)|(1<<10)
    m.mav.set_position_target_local_ned_send(
        int(time.time()*1000) & 0xFFFFFFFF,
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,
        float(vx), float(vy), float(vz),
        0, 0, 0,
        0, float(yaw_rate)
    )

def guided_hold_async():
    def _worker():
        try:
            m, mavutil = _connect_mav()
            _set_mode_guided(m, mavutil)
            print(f"[MAV] HOLD {HOLD_DURATION_S}s @ {HOLD_RATE_HZ}Hz")
            t_end = time.time() + HOLD_DURATION_S
            dt = 1.0 / float(max(1, HOLD_RATE_HZ))
            while time.time() < t_end:
                _send_local_vel(m, mavutil, 0.0, 0.0, 0.0)
                time.sleep(dt)
            print("[MAV] HOLD done.")
        except Exception as e:
            print(f"[MAV] guided_hold error: {e}")
        finally:
            try: m.close()
            except: pass
    threading.Thread(target=_worker, daemon=True).start()

def guided_descend_to_relalt_async(target_m=TARGET_REL_ALT_M, speed_mps=DESCEND_SPEED_MPS,
                                   timeout_s=DESCEND_TIMEOUT_S, reach_eps=REACH_EPS_M):
    """
    Position setpoint 방식: '상대고도=target_m'을 지속 전송 (MAV_FRAME_GLOBAL_RELATIVE_ALT_INT).
    x/y(위도/경도)는 현재 위치 유지, z만 target_m로 설정.
    """
    def _worker():
        try:
            m, mavutil = _connect_mav()

            # (추가) GLOBAL_POSITION_INT 메시지 스트림 요청: 5 Hz
            print("[MAV] Requesting GLOBAL_POSITION_INT stream at 5Hz")
            try:
                m.mav.command_long_send(
                    m.target_system, m.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                    mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                    200000,  # microseconds (5Hz)
                    0, 0, 0, 0, 0
                )
                time.sleep(0.5)
            except Exception as e:
                print(f"[MAV] Stream request failed (continuing): {e}")

            _set_mode_guided(m, mavutil)

            target = float(target_m)
            print(f"[MAV] DESCEND (ALT SETPOINT) to rel_alt {target:.2f} m")

            # (유지) POSITION 전용 type_mask: 속도/가속/yaw/yaw_rate 무시
            # 안전안: 비트 3,4,5,6,7,8,10,11
            TYPE_MASK_POS_ONLY = (
                (1<<3)|(1<<4)|(1<<5)|    # ignore vx, vy, vz
                (1<<6)|(1<<7)|(1<<8)|    # ignore ax, ay, az
                (1<<10)|(1<<11)          # ignore yaw, yaw_rate
            )
            # force setpoint을 명시적으로 무시하려면 아래 라인 해제(대부분 불필요)
            # TYPE_MASK_POS_ONLY |= (1<<9)

            t0 = time.time()
            last_print = 0.0
            last_lat = None
            last_lon = None

            while time.time() - t0 < timeout_s:
                msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.5)
                rel_m = None
                if msg:
                    if hasattr(msg, "relative_alt"):
                        rel_m = msg.relative_alt / 1000.0  # mm → m
                    if hasattr(msg, "lat"): last_lat = int(msg.lat)
                    if hasattr(msg, "lon"): last_lon = int(msg.lon)
                    
                     ### [추가] 수신된 메시지 매번 터미널에 출력
                    print(f"[MAV][STREAM] lat={msg.lat}, lon={msg.lon}, rel_alt={msg.relative_alt/1000.0:.2f} m")
                    
                else:
                    print("[MAV] No GLOBAL_POSITION_INT received - check stream config")

                # 위도/경도 모르면 읽힐 때까지 정지 명령
                if last_lat is None or last_lon is None:
                    _send_local_vel(m, mavutil, 0.0, 0.0, 0.0)
                    continue

                if time.time() - last_print > 1.0 and rel_m is not None:
                    print(f"[MAV] rel_alt={rel_m:.2f} m → target {target:.2f} m (pos-hold)")
                    last_print = time.time()

                # 목표치 도달 체크
                if rel_m is not None and rel_m <= target + reach_eps:
                    print("[MAV] Target reached → HOLD")
                    for _ in range(10):  # 약 2초간 정지
                        _send_local_vel(m, mavutil, 0.0, 0.0, 0.0)
                        time.sleep(0.2)
                    break

                # 현재 위도/경도 유지 + 목표 상대고도(target) 세팅
                m.mav.set_position_target_global_int_send(
                    int(time.time()*1000) & 0xFFFFFFFF,
                    m.target_system, m.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    TYPE_MASK_POS_ONLY,
                    int(last_lat), int(last_lon), float(target),
                    0.0, 0.0, 0.0,     # vx,vy,vz 무시
                    0.0, 0.0, 0.0,     # accel 무시
                    0.0, 0.0           # yaw, yaw_rate 무시
                )

                time.sleep(0.2)  # ≈5Hz로 목표 반복 전송

            else:
                print("[MAV] Timeout → HOLD")
                for _ in range(10):
                    _send_local_vel(m, mavutil, 0.0, 0.0, 0.0)
                    time.sleep(0.2)

        except Exception as e:
            print(f"[MAV] guided_descend error: {e}")
        finally:
            try: m.close()
            except: pass
    threading.Thread(target=_worker, daemon=True).start()

# ===================== 기타 유틸 =====================
class Ticker:
    def __init__(self, interval):
        self.interval = float(interval)
        self._next_tick = time.time() + self.interval
    def wait(self):
        now = time.time()
        sleep_time = self._next_tick - now
        if sleep_time > 0: time.sleep(sleep_time)
        else: self._next_tick = now
        self._next_tick += self.interval

def _init_cuda_engine():
    cuda.init()
    ctx = cuda.Device(0).make_context()
    engine = load_engine(ENGINE_PATH)
    n, c, h, w, in_binding_idx = get_engine_input_shape(engine)
    return ctx, engine, engine.create_execution_context(), h, w, in_binding_idx

def create_pipelines():
    gpu_executor = ThreadPoolExecutor(max_workers=1)
    ctx, engine, context, net_h, net_w, in_binding_idx = gpu_executor.submit(_init_cuda_engine).result()
    gpu_lock = threading.Lock()
    p_wait = Pipeline(gpu_lock, context, gpu_executor, engine, net_h, net_w, in_binding_idx)
    p_next = Pipeline(gpu_lock, context, gpu_executor, engine, net_h, net_w, in_binding_idx)
    p_wait.ensure_ready(); p_next.ensure_ready()
    return ctx, (p_wait, p_next), net_h, net_w

# ===================== 입력 오픈 (GStreamer) =====================
def open_webcam_gst(path, width, height, fps, retries=6, delay=1.0):
    # /dev/video0 → BGR 프레임
    gst = (
        f'v4l2src device={path} ! '
        f'video/x-raw,framerate={fps}/1 ! '
        f'videoscale ! videoconvert ! '
        f'video/x-raw,width={width},height={height},format=BGR ! '
        f'appsink drop=true max-buffers=2'
    )
    for i in range(retries):
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened(): return cap
        print(f"⚠️  웹캠 열기 재시도({i+1}/{retries})… {path} 점유/초기화 지연 가능")
        time.sleep(delay)
    return None

# ===================== Main =====================
if __name__ == "__main__":
    ctx = None
    cap = None
    out = None
    try:
        # 입력: /dev/video0
        cap = open_webcam_gst(DEVICE_PATH, COLOR_WIDTH, COLOR_HEIGHT, FPS)
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"❌ 웹캠 열기 실패 ({DEVICE_PATH} 확인)")

        # 출력: web_cam_1080.py와 동일한 RTP/UDP 송출 체인
        gst_out = (
            'appsrc is-live=true block=true format=GST_FORMAT_TIME do-timestamp=true ! '
            f'video/x-raw,format=BGR,width={COLOR_WIDTH},height={COLOR_HEIGHT},framerate={FPS}/1 ! '
            'queue leaky=2 ! videoconvert ! video/x-raw,format=BGRx ! '
            'nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! '
            f'nvv4l2h264enc iframeinterval={FPS} bitrate=700000 insert-sps-pps=true preset-level=1 ! '
            'h264parse config-interval=1 ! rtph264pay pt=96 config-interval=1 ! '
            f'udpsink host={GROUND_IP} port={PORT} sync=false'
        )
        out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, FPS, (COLOR_WIDTH, COLOR_HEIGHT))
        if not out.isOpened():
            raise RuntimeError("❌ GStreamer VideoWriter OPEN 실패(송출 파이프라인 생성 실패)")

        print(f"✅ UDP H.264 RTP 송신 준비 완료 → {GROUND_IP}:{PORT}")

        # YOLO 엔진/파이프라인
        ctx, (p_wait, p_next), net_h, net_w = create_pipelines()
        print(f"📐 Engine input size: {net_w}x{net_h}")

        # K-of-N 투표 관련
        N = max(1, int(FPS * WINDOW_SEC))
        hits = deque(maxlen=N)
        trigger_on = False
        triggered_at = 0.0
        sustain_counter = 0
        ticker = Ticker(1 / FPS)
        t_prev = time.time()

        # Priming
        ret0, img0 = cap.read()
        if not ret0:
            raise RuntimeError("❌ 첫 프레임을 가져오지 못했습니다.")
        executor = ThreadPoolExecutor(max_workers=2)
        p_wait.submit(executor, img0.copy())

        # ===== 메인 루프 =====
        while True:
            ticker.wait()
            ret, img = cap.read()
            if not ret:
                continue

            # 다음 프레임 제출
            p_next.submit(executor, img.copy())

            # 이전 제출 결과 수신
            result = p_wait.wait()
            if result is None:
                continue
            final_img, dets = result

            # 파이프라인 스왑
            p_wait, p_next = p_next, p_wait

            # YOLO hit 판정
            hit = False
            if dets:
                max_conf = max(d[-1] for d in dets)
                if max_conf >= HIT_CONF_MIN:
                    hit = True
            hits.append(1 if hit else 0)

            # 비율 계산
            k = sum(hits)
            ratio = k / float(len(hits)) if hits else 0.0

            # 상태 JSON (디버그용)
            try:
                status = {
                    "ts": time.time(),
                    "window": len(hits),
                    "hits": int(k),
                    "ratio": ratio,
                    "fps": FPS,
                    "threshold": TRIGGER_RATIO,
                    "conf_min": HIT_CONF_MIN,
                    "trigger_on": trigger_on,
                    "action": ACTION_MODE
                }
                tmp = STATUS_JSON + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(status, f)
                os.replace(tmp, STATUS_JSON)
            except Exception:
                pass

            # 트리거 판정
            now = time.time()
            if ratio >= TRIGGER_RATIO:
                sustain_counter += 1
            else:
                sustain_counter = 0

            sustained   = (sustain_counter / FPS) >= SUSTAIN_SEC
            cooldown_ok = (now - triggered_at) >= COOLDOWN_SEC

            if (not trigger_on) and sustained and cooldown_ok:
                print(f"[TRIGGER] ratio={ratio:.2f}  window={len(hits)}  k={k}  → ACTION={ACTION_MODE}")
                trigger_on = True
                triggered_at = now
                # === 선택 동작 수행 (GUIDED 전환 포함) ===
                if ACTION_MODE == "hold":
                    guided_hold_async()
                elif ACTION_MODE == "descend5":
                    guided_descend_to_relalt_async(TARGET_REL_ALT_M, DESCEND_SPEED_MPS, DESCEND_TIMEOUT_S, REACH_EPS_M)
                else:
                    print(f"[WARN] Unknown ACTION_MODE: {ACTION_MODE}")

            if trigger_on and (now - triggered_at) >= COOLDOWN_SEC:
                trigger_on = False
                sustain_counter = 0

            # 오버레이 + 송출
            if final_img is not None:
                t_now = time.time()
                fps_now = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now
                cv2.putText(final_img, f"FPS:{fps_now:.2f}  k={k}/{len(hits)} ({ratio*100:.0f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(final_img)

    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 종료 요청 – 스트리밍 중단")
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
    finally:
        print("🧹 정리 및 종료…")
        try:
            if out is not None and out.isOpened(): out.release()
        except: pass
        try:
            if cap is not None and cap.isOpened(): cap.release()
        except: pass
        try:
            if ctx is not None: ctx.pop()
        except: pass

