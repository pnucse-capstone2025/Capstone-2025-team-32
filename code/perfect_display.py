#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hold_descend5_webcam_mode.py
- 입력: /dev/video0 (GStreamer v4l2src + appsink)
- 추론: YOLOv5 TensorRT
- 송출: NVENC → H.264 → RTP/UDP → udpsink  (지상: ffplay -i color.sdp)
- 트리거: 최근 2초 프레임 중 70% 이상 검출 시 AUTO→GUIDED 전환 후 ACTION_MODE 실행
- 표시: 우측 상단에 실제 Pixhawk 비행모드(MAVLink HEARTBEAT) 실시간 표시
      + 코드가 모드 전환을 보낼 때는 즉시 "GUIDED" 강제 표시(쿨다운 동안)
- 안정성: 카메라/스트리밍 오류 시 자동 재연결
"""

import os, time, json, threading, cv2, numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import tensorrt as trt
import pycuda.driver as cuda
from utils.plots import Annotator, colors

# ===================== 선택 동작 =====================
# "hold" 또는 "descend5"
ACTION_MODE = "hold"

# ===================== 사용자 설정 =====================
GROUND_IP    = "10.0.0.4"        # 지상 PC IP
PORT         = 5000
ENGINE_PATH  = "runs/train/1490_images_1280_b164/weights/best_int8.engine"
CONF_THR     = 0.30
IOU_THR      = 0.45
FPS          = 15

COLOR_WIDTH, COLOR_HEIGHT = 1280, 720
DEVICE_PATH  = "/dev/video0"
NAMES = ["Basket"]

WINDOW_SEC      = 2.0
HIT_CONF_MIN    = 0.50
TRIGGER_RATIO   = 0.70
SUSTAIN_SEC     = 0.4
COOLDOWN_SEC    = 8.0
STATUS_JSON     = "/tmp/yolo_kofn.json"

# MAVLink 연결 (시리얼 또는 UDP; 필요 시 아래 한 줄만 바꾸세요)
SERIAL_PORT     = "/dev/ttyACM0"   # 예) UDP 포워딩이면 "udp:0.0.0.0:14550"
SERIAL_BAUD     = 115200
HOLD_DURATION_S = 1.0
HOLD_RATE_HZ    = 5
TARGET_REL_ALT_M  = 5.0
DESCEND_SPEED_MPS = 0.6
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

# ===================== 모드 표시: HEARTBEAT + 강제표시 =====================
_current_mode = "N/A"          # HEARTBEAT가 준 실제 모드
_mode_lock    = threading.Lock()

_forced_mode_text  = None      # 예: "GUIDED"
_forced_mode_until = 0.0       # time.time() 기준 만료 시각
_forced_lock       = threading.Lock()

def set_forced_mode(text: str, duration_sec: float):
    """표시용 모드를 duration_sec 동안 강제로 보여줌(HEARTBEAT와 무관)."""
    global _forced_mode_text, _forced_mode_until
    with _forced_lock:
        _forced_mode_text  = text
        _forced_mode_until = time.time() + float(duration_sec)

def get_mode_text_to_show():
    """오버레이에 표시할 모드 문자열(강제표시가 우선, 만료되면 HEARTBEAT)."""
    now = time.time()
    with _forced_lock:
        if _forced_mode_text and now < _forced_mode_until:
            return _forced_mode_text
    with _mode_lock:
        return _current_mode

class ModeWatcher(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self._stop = threading.Event()

    def run(self):
        global _current_mode
        try:
            from pymavlink import mavutil
            m = mavutil.mavlink_connection(self.port, baud=self.baud)
            m.wait_heartbeat(timeout=5)
            while not self._stop.is_set():
                msg = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1.0)
                if msg:
                    mode = mavutil.mode_string_v10(msg)
                    if mode:
                        with _mode_lock:
                            _current_mode = mode
        except Exception:
            with _mode_lock:
                _current_mode = "N/A"

    def stop(self):
        self._stop.set()

# ===================== MAVLink 동작 =====================
def _connect_mav():
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(SERIAL_PORT, baud=SERIAL_BAUD)
    m.wait_heartbeat(timeout=5)
    return m, mavutil

def _set_mode_guided(m, mavutil):
    # 표시 즉시 반영(쿨다운 동안)
    set_forced_mode("GUIDED", COOLDOWN_SEC)
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
        except Exception as e:
            print(f"[MAV] guided_hold error: {e}")
        finally:
            try: m.close()
            except: pass
    threading.Thread(target=_worker, daemon=True).start()

def guided_descend_to_relalt_async(target_m=TARGET_REL_ALT_M, speed_mps=DESCEND_SPEED_MPS,
                                   timeout_s=DESCEND_TIMEOUT_S, reach_eps=REACH_EPS_M):
    def _worker():
        try:
            m, mavutil = _connect_mav()
            _set_mode_guided(m, mavutil)
            target = float(target_m)
            print(f"[MAV] DESCEND to rel_alt {target:.2f} m, speed {speed_mps:.2f} m/s")
            t0 = time.time()
            last_print = 0.0
            while time.time() - t0 < timeout_s:
                msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.5)
                rel_m = msg.relative_alt/1000.0 if (msg and hasattr(msg, "relative_alt")) else None
                if rel_m is None:
                    _send_local_vel(m, mavutil, 0.0, 0.0, 0.0); continue
                if time.time() - last_print > 1.0:
                    print(f"[MAV] rel_alt={rel_m:.2f} m → target {target:.2f} m"); last_print = time.time()
                if rel_m <= target + reach_eps:
                    print("[MAV] Target reached → HOLD")
                    for _ in range(10): _send_local_vel(m, mavutil, 0,0,0); time.sleep(0.2)
                    break
                _send_local_vel(m, mavutil, 0.0, 0.0, abs(speed_mps))  # +Vz 하강
                time.sleep(0.2)
            else:
                print("[MAV] Timeout → HOLD")
                for _ in range(10): _send_local_vel(m, mavutil, 0,0,0); time.sleep(0.2)
        except Exception as e:
            print(f"[MAV] guided_descend error: {e}")
        finally:
            try: m.close()
            except: pass
    threading.Thread(target=_worker, daemon=True).start()

# ===================== 입력/출력 파이프라인 =====================
def open_webcam_gst(path, width, height, fps, retries=6, delay=1.0):
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

def open_udpsink_gst(width, height, fps):
    gst_out = (
        'appsrc is-live=true block=true format=GST_FORMAT_TIME do-timestamp=true ! '
        f'video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! '
        'queue leaky=2 ! videoconvert ! video/x-raw,format=BGRx ! '
        'nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! '
        f'nvv4l2h264enc iframeinterval={fps} bitrate=700000 insert-sps-pps=true preset-level=1 ! '
        'h264parse config-interval=1 ! rtph264pay pt=96 config-interval=1 ! '
        f'udpsink host={GROUND_IP} port={PORT} sync=false'
    )
    return cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (width, height))

# ===================== Main =====================
if __name__ == "__main__":
    ctx = None
    cap = None
    out = None
    mode_watcher = None
    try:
        # 모드 워처 시작(실패해도 메인 영향 X)
        mode_watcher = ModeWatcher(SERIAL_PORT, SERIAL_BAUD)
        mode_watcher.start()

        # 입력/출력
        cap = open_webcam_gst(DEVICE_PATH, COLOR_WIDTH, COLOR_HEIGHT, FPS)
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"❌ 웹캠 열기 실패 ({DEVICE_PATH})")
        out = open_udpsink_gst(COLOR_WIDTH, COLOR_HEIGHT, FPS)
        if not out.isOpened():
            raise RuntimeError("❌ GStreamer VideoWriter OPEN 실패")
        print(f"✅ UDP H.264 RTP 송신 준비 완료 → {GROUND_IP}:{PORT}")

        # YOLO 엔진/파이프라인
        def _init_cuda_engine():
            cuda.init()
            c = cuda.Device(0).make_context()
            eng = load_engine(ENGINE_PATH)
            n, cch, hh, ww, in_idx = get_engine_input_shape(eng)
            return c, eng, eng.create_execution_context(), hh, ww, in_idx

        gpu_executor = ThreadPoolExecutor(max_workers=1)
        ctx, engine, context, net_h, net_w, in_binding_idx = gpu_executor.submit(_init_cuda_engine).result()
        gpu_lock = threading.Lock()
        p_wait = Pipeline(gpu_lock, context, gpu_executor, engine, net_h, net_w, in_binding_idx)
        p_next = Pipeline(gpu_lock, context, gpu_executor, engine, net_h, net_w, in_binding_idx)
        p_wait.ensure_ready(); p_next.ensure_ready()
        print(f"📐 Engine input size: {net_w}x{net_h}")

        # 투표/표시 변수
        N = max(1, int(FPS * WINDOW_SEC))
        hits = deque(maxlen=N)
        trigger_on = False
        triggered_at = 0.0
        sustain_counter = 0
        t_prev = time.time()
        executor = ThreadPoolExecutor(max_workers=2)

        # 첫 프레임
        ret0, img0 = cap.read()
        if not ret0: raise RuntimeError("❌ 첫 프레임 획득 실패")
        p_wait.submit(executor, img0.copy())

        while True:
            ret, img = cap.read()
            if not ret:
                # 카메라 재연결
                print("⚠️ 프레임 읽기 실패 → 카메라 재연결")
                try: cap.release()
                except: pass
                cap = open_webcam_gst(DEVICE_PATH, COLOR_WIDTH, COLOR_HEIGHT, FPS)
                continue

            p_next.submit(executor, img.copy())
            result = p_wait.wait()
            if result is None: continue
            final_img, dets = result
            p_wait, p_next = p_next, p_wait

            # YOLO hit
            hit = any(d[-1] >= HIT_CONF_MIN for d in dets)
            hits.append(1 if hit else 0)
            k = sum(hits)
            ratio = k / float(len(hits)) if hits else 0.0

            now = time.time()
            if ratio >= TRIGGER_RATIO: sustain_counter += 1
            else:                       sustain_counter  = 0

            sustained   = (sustain_counter / FPS) >= SUSTAIN_SEC
            cooldown_ok = (now - triggered_at) >= COOLDOWN_SEC

            if (not trigger_on) and sustained and cooldown_ok:
                print(f"[TRIGGER] ratio={ratio:.2f} k={k}/{len(hits)} → ACTION={ACTION_MODE}")
                trigger_on = True
                triggered_at = now
                # 화면에는 즉시 GUIDED로 강제 표시(HEARTBEAT 기다리지 않음)
                set_forced_mode("GUIDED", COOLDOWN_SEC)
                if ACTION_MODE == "hold":
                    guided_hold_async()
                elif ACTION_MODE == "descend5":
                    guided_descend_to_relalt_async(TARGET_REL_ALT_M, DESCEND_SPEED_MPS, DESCEND_TIMEOUT_S, REACH_EPS_M)

            if trigger_on and (now - triggered_at) >= COOLDOWN_SEC:
                trigger_on = False
                sustain_counter = 0

            # ── 오버레이 & 송출 ──
            if final_img is not None:
                # 좌상단: FPS / k-of-N
                t_now = time.time()
                fps_now = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now
                cv2.putText(final_img, f"FPS:{fps_now:.2f}  k={k}/{len(hits)} ({ratio*100:.0f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # 우측상단: 모드 표시 (강제표시가 우선, 만료되면 HEARTBEAT 값)
                mode_text = f"MODE: {get_mode_text_to_show()}"
                (tw, th), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                x = COLOR_WIDTH - tw - 10
                y = 40
                mt = mode_text.upper()
                if "GUIDED" in mt:
                    color = (0,0,255)
                elif "AUTO" in mt:
                    color = (255,0,0)
                else:
                    color = (0,255,0)
                cv2.putText(final_img, mode_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                try:
                    out.write(final_img)
                except Exception as e:
                    print(f"⚠️ 송출 오류: {e} → 파이프라인 재생성")
                    try: out.release()
                    except: pass
                    out = open_udpsink_gst(COLOR_WIDTH, COLOR_HEIGHT, FPS)

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
    finally:
        try:
            if out is not None and out.isOpened(): out.release()
        except: pass
        try:
            if cap is not None and cap.isOpened(): cap.release()
        except: pass
        try:
            if mode_watcher is not None: mode_watcher.stop()
        except: pass
        try:
            if ctx is not None: ctx.pop()
        except: pass

