#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1440_70%.py  (fixed double-buffer + safe init)
- RealSense COLOR only
- TensorRT YOLO 추론 결과: 최근 WINDOW_SEC 동안의 프레임 중 HIT 비율 ≥ TRIGGER_RATIO(예: 70%)
  이면 하강 스크립트 한 번 실행
- 첫 프레임 priming 후 더블버퍼로 안정 동작
"""

import os, sys, time, json, subprocess, threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import pycuda.driver as cuda

from utils.plots import Annotator, colors

# ===================== 사용자 설정 =====================
GROUND_IP    = "10.0.0.4"
PORT         = 5000

# 모델 / 해상도
ENGINE_PATH  = "runs/train/1490_images_1280_b164/weights/best_int8.engine"  # 실제 경로로 변경
CONF_THR     = 0.30
IOU_THR      = 0.45
FPS          = 15
COLOR_WIDTH, COLOR_HEIGHT = 1280, 720
NAMES = ["Basket"]  # 단일 클래스 가정

# 트리거 파라미터
WINDOW_SEC      = 2.0          # 투표 윈도우(초)
HIT_CONF_MIN    = 0.50         # YOLO conf 최소
TRIGGER_RATIO   = 0.70         # k-of-n 임계치 (70%)
SUSTAIN_SEC     = 0.4          # 임계 충족 지속 시간
COOLDOWN_SEC    = 8.0          # 재발동 쿨다운
STATUS_JSON     = "/tmp/yolo_kofn.json"

# 하강 스크립트(비동기 실행 1회)
DESCENDER = [
    "python3", "down.py",
    "--serial", "/dev/ttyACM0",
    "--baud", "115200",
    "--vz", "0.4",           # +0.4 m/s 하강(NED 기준)
    "--duration", "6.0",
    "--brake_ms", "500",
    "--resume-auto"
]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ===================== TensorRT 유틸 =====================
def load_engine(path: str):
    print(f"🚀 Loading TensorRT engine from: {path}")
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("❌ TensorRT 엔진 로드 실패")
    print("✅ TensorRT Engine Loaded Successfully.")
    return engine

def get_engine_input_shape(engine):
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            shape = list(engine.get_binding_shape(i))
            if shape[0] < 0:
                shape[0] = 1
            if len(shape) == 4:
                n, c, h, w = shape
                return int(n), int(c), int(h), int(w), i
            raise RuntimeError(f"Unexpected input shape: {shape}")
    raise RuntimeError("No input binding found")

def allocate_buffers(engine):
    host_in = device_in = None
    host_out, device_out, bindings = [], [], []
    stream = cuda.Stream()
    for i in range(engine.num_bindings):
        shape = list(engine.get_binding_shape(i))
        if shape[0] < 0:
            shape[0] = 1
        size  = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(i):
            host_in, device_in = host_mem, device_mem
        else:
            host_out.append(host_mem)
            device_out.append(device_mem)
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
    if x.size == 0:
        return out
    mask = x[:, 4] > conf_thres
    x = x[mask]
    if not x.size:
        return out
    boxes = xywh2xyxy(x[:, :4])
    scores = x[:, 4]
    cls = np.zeros_like(scores)
    b4nms = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
    idxs = cv2.dnn.NMSBoxes(b4nms, scores.tolist(), conf_thres, iou_thres)
    if hasattr(idxs, "flatten"):
        idxs = idxs.flatten()
        final = np.concatenate((boxes[idxs], scores[idxs, None], cls[idxs, None]), 1)
        out[0] = final.astype(np.float32)
    return out

def scale_coords(img1_shape, coords, img0_shape):
    gain_w = img1_shape[1] / img0_shape[1]
    gain_h = img1_shape[0] / img0_shape[0]
    coords[:, [0, 2]] /= gain_w
    coords[:, [1, 3]] /= gain_h
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    return coords

# ===================== 파이프라인 =====================
class Pipeline:
    def __init__(self, gpu_lock, context, executor, engine, net_h, net_w, in_binding_idx):
        self.gpu_lock = gpu_lock
        self.context  = context
        self.gpu_exec = executor  # GPU 전용 executor (max_workers=1)
        self.net_h, self.net_w = net_h, net_w
        self.in_binding_idx = in_binding_idx

        # 프레임 버퍼
        self.img_rs      = np.empty((self.net_h, self.net_w, 3), dtype=np.uint8)
        self.img_rgb     = np.empty((self.net_h, self.net_w, 3), dtype=np.uint8)
        self.img_rgb_f32 = np.empty((self.net_h, self.net_w, 3), dtype=np.float32)

        # 초기화 작업(버퍼 할당)을 먼저 제출하고 완료될 때까지 기다리게 함
        self._init_future = self.gpu_exec.submit(self._init_buffers, engine)
        self._run_future = None  # 이후 run() 제출 결과

    def _init_buffers(self, engine):
        (self.h_in, self.d_in), (self.h_outs, self.d_outs, self.bindings, self.stream) = allocate_buffers(engine)

    def ensure_ready(self):
        # 초기화 작업이 끝날 때까지 블로킹
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
        # CPU 스레드풀에서 run() 실행
        self._run_future = executor.submit(Pipeline.run, self, img0)

    def wait(self):
        # run() 제출된 경우에만 결과 반환
        if self._run_future is None:
            return None
        res = self._run_future.result()
        self._run_future = None
        return res

# ===================== 기타 유틸 =====================
class Ticker:
    def __init__(self, interval):
        self.interval = float(interval)
        self._next_tick = time.time() + self.interval
    def wait(self):
        now = time.time()
        sleep_time = self._next_tick - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            self._next_tick = now
        self._next_tick += self.interval

def _init_cuda_engine():
    cuda.init()
    ctx = cuda.Device(0).make_context()
    engine = load_engine(ENGINE_PATH)
    n, c, h, w, in_binding_idx = get_engine_input_shape(engine)
    return ctx, engine, engine.create_execution_context(), h, w, in_binding_idx

def create_pipelines():
    gpu_executor = ThreadPoolExecutor(max_workers=1)  # GPU job 전용
    ctx, engine, context, net_h, net_w, in_binding_idx = gpu_executor.submit(_init_cuda_engine).result()
    gpu_lock = threading.Lock()
    p_wait = Pipeline(gpu_lock, context, gpu_executor, engine, net_h, net_w, in_binding_idx)
    p_next = Pipeline(gpu_lock, context, gpu_executor, engine, net_h, net_w, in_binding_idx)
    # 두 파이프라인 모두 버퍼 초기화가 끝나도록 보장
    p_wait.ensure_ready()
    p_next.ensure_ready()
    return ctx, (p_wait, p_next), net_h, net_w

# ===================== 메인 =====================
if __name__ == "__main__":
    ctx = None
    pipe = None
    out = None
    try:
        # RealSense COLOR만 사용
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        profile = pipe.start(cfg)

        # UDP H.264 송출
        gst = (
            f'appsrc ! videoconvert ! video/x-raw,format=BGRx ! '
            f'nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! '
            f'nvv4l2h264enc iframeinterval={FPS} bitrate=700000 insert-sps-pps=true ! '
            f'video/x-h264,stream-format=byte-stream ! '
            f'rtph264pay config-interval=1 pt=96 ! '
            f'udpsink host={GROUND_IP} port={PORT} sync=false'
        )
        out = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, FPS, (COLOR_WIDTH, COLOR_HEIGHT))
        if not out.isOpened():
            raise RuntimeError("❌ GStreamer VideoWriter OPEN 실패")

        # CUDA/엔진/파이프라인
        ctx, (p_wait, p_next), net_h, net_w = create_pipelines()
        print(f"📐 Engine input size: {net_w}x{net_h}")

        # k-of-n 투표
        N = max(1, int(FPS * WINDOW_SEC))
        hits = deque(maxlen=N)
        trigger_on = False
        triggered_at = 0.0
        sustain_counter = 0
        ticker = Ticker(1 / FPS)
        t_prev = time.time()

        # ===== Priming: 첫 프레임을 p_wait에 먼저 제출 =====
        frames0 = pipe.wait_for_frames()
        cfrm0 = frames0.get_color_frame()
        if not cfrm0:
            raise RuntimeError("❌ RealSense 첫 프레임을 가져오지 못했습니다.")
        img0 = np.ascontiguousarray(cfrm0.get_data())
        executor = ThreadPoolExecutor(max_workers=2)  # CPU측 파이프라인 제어
        p_wait.submit(executor, img0.copy())

        # ===== 메인 루프 (더블버퍼) =====
        while True:
            ticker.wait()
            frames = pipe.wait_for_frames()
            cfrm = frames.get_color_frame()
            if not cfrm:
                continue
            img = np.ascontiguousarray(cfrm.get_data())

            # 다음 프레임 제출
            p_next.submit(executor, img.copy())

            # 이전 제출 결과 수신
            result = p_wait.wait()
            if result is None:
                # 안전 가드 (논리상 발생하지 않게 했지만 혹시나 대비)
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

            # k-of-n 상태
            k = sum(hits)
            ratio = k / float(len(hits)) if hits else 0.0

            # 상태 JSON (디버그)
            try:
                status = {
                    "ts": time.time(),
                    "window": len(hits),
                    "hits": int(k),
                    "ratio": ratio,
                    "fps": FPS,
                    "threshold": TRIGGER_RATIO,
                    "conf_min": HIT_CONF_MIN,
                    "trigger_on": trigger_on
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
                print(f"[TRIGGER] ratio={ratio:.2f}  window={len(hits)}  k={k}")
                trigger_on = True
                triggered_at = now
                try:
                    subprocess.Popen(DESCENDER, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    print(f"[RUN] {' '.join(DESCENDER)}")
                except Exception as e:
                    print(f"[ERR] down.py 실행 실패: {e}")

            if trigger_on and (now - triggered_at) >= COOLDOWN_SEC:
                trigger_on = False
                sustain_counter = 0

            # 오버레이 및 송출
            if final_img is not None:
                t_now = time.time()
                dt = t_now - t_prev
                fps_now = 1.0 / dt if dt > 0 else 0.0
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
            if out is not None and out.isOpened():
                out.release()
        except:
            pass
        try:
            if pipe is not None:
                pipe.stop()
        except:
            pass
        try:
            if ctx is not None:
                ctx.pop()  # PyCUDA 컨텍스트 누수 방지
        except:
            pass

