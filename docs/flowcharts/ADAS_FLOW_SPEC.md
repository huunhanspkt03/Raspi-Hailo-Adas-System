# ADAS Flow Spec

Tài liệu này là contract giữa flowchart và code. Bộ flowchart hiện tại ưu tiên
kiến trúc Python runtime theo hướng multiple-threading, nhưng vẫn giữ single
loop như đường replay/debug.

## Internet References Used

- GStreamer queue design: `queue` được dùng như ranh giới buffering và tách
  thread giữa các stage pipeline.
  <https://gstreamer.freedesktop.org/documentation/coreelements/queue.html>
- ROS 2 executor/callback-group model: tham khảo cách nhóm callback để kiểm
  soát phần nào được chạy song song, phần nào cần mutual exclusion.
  <https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Executors.html>
- Python `queue`: chuẩn producer/consumer multi-thread an toàn cho trao đổi
  dữ liệu giữa worker.
  <https://docs.python.org/3/library/queue.html>
- Hailo async inference examples/docs: tham khảo hướng tiếp theo cho backend
  không khóa dài trong lane/object workers.
  <https://github.com/hailo-ai/Hailo-Application-Code-Examples>

## Design Principles

- Latency-first: queue nhỏ, drop stale frame/result, không tích backlog.
- Flowchart-first readability: generator tự nới canvas theo node/edge ngoài
  cùng và cộng margin, để khi mở bằng diagrams.net không bị sát mép trang.
- Task-block layout: mỗi khối trong flowchart chỉ mô tả nhiệm vụ chính, không
  ghi chi tiết câu lệnh/hàm; chi tiết code nằm ở spec hoặc module tương ứng.
- Wire clarity: dây đi theo lane trái-phải hoặc trên-dưới; hạn chế dây vòng
  dài, dây phụ và nhánh chồng lên nhau.
- Fan-out rõ ràng: capture đưa cùng `FramePacket` vào `lane_q` và `object_q`;
  hai nhánh không tranh nhau frame.
- Ownership rõ ràng: worker nào giữ state của worker đó. Downstream chỉ đọc
  packet snapshot, không đọc private tracker/localizer state.
- Synchronization rõ ràng: fusion ghép theo `frame_id`; nếu lệch quá
  `STALE_FRAME_TOLERANCE`, packet cũ bị evict.
- Backend hiện tại guarded bằng shared `infer_lock`; roadmap là async Hailo
  hoặc executor/callback-group style scheduling khi backend đã an toàn.

## State Contracts

- `FramePacket`: `frame_id`, `timestamp`, `frame`, `frame_h`, `frame_w`
- `LanePacket`: `frame_id`, `timestamp`, `lane_state`, `source_frame`,
  `lane_area`, `semantics`, `class_probs`, `lane_path`, `lane_bounds`,
  `corridor`
- `ObjectPacket`: `frame_id`, `timestamp`, `source_frame`, `detections`,
  `front_object`
- `BehaviorPacket`: `frame_id`, `timestamp`, `frame`, `lane`, `objects`,
  `behavior_action`, `behavior_reason`, `control`
- `ControlCommand`: `steering`, `throttle`, `rc_enabled`, `safe`
- Future `HealthState`: `camera_ok`, `backend_ok`, `fps_ok`, `queue_drops`,
  `stale_packets`, `degraded_mode`, `stop_required`

## Flow -> Module Goal

### 01. Tổng thể ADAS mini

- Input: CLI args, YAML config, camera/video source
- Output: runtime chạy sạch hoặc stop sạch
- Mục tiêu: mô tả top-level multi-thread path từ init đến cleanup.

### 02. Luồng xử lý một frame

- Input: một frame hợp lệ
- Output: `BehaviorPacket` được render/actuate
- Cần có: `FramePacket.make`, fan-out queue, lane/object workers, fusion by
  `frame_id`, output loop.

### 03. Lane worker

- Input: `FramePacket`
- Output: `LanePacket`
- Cần có: preprocess YOLO26s_seg, locked inference, decode mask/lane_area,
  boundary tracker, red center, dynamic/render corridor, `LaneState`.

### 04. Lane/corridor geometry

- Input: `lane_area`, `semantics`, `boundary_track`, `red_center`, frame size
- Output: corridor/guidance dict cho overlay và object localization
- Lưu ý: runtime hiện tại không chạy BEV/IPM; flow này mô tả geometry active:
  purple model-detected, red ego-anchored, yellow fixed-width corridor.

### 05. Lane tracking theo thời gian

- Input: current boundary candidates + previous tracker state + width memories
- Output: `lane_bounds`
- Cần có: validation trước EMA, purple width memory, lane-area width learner,
  resolve policy, source/diagnostic fields.

### 06. Điều khiển

- Input: matched `FramePacket`, optional `LanePacket`, optional `ObjectPacket`
- Output: `BehaviorPacket`
- Cần có: steering compute/gate/rate-limit/smooth, behavior throttle, default
  action theo runtime mode, `ControlCommand`.

### 07. Object worker

- Input: `FramePacket`
- Output: `ObjectPacket`
- Cần có: preprocess YOLO26s, locked inference, decode, ByteTrack,
  `ObjectLocalizer.localize`, traffic-light group fusion, `front_object`.

### 08. Behavior logic

- Input: detections đã enrich pose, control config, frame width
- Output: `BehaviorState`
- Priority: red light stop > leading vehicle follow > speed cap > go.

### 09. Runtime lifecycle

- Input: CLI args + YAML config
- Output: single-loop hoặc threaded runtime started, rồi cleanup sạch
- Cần có: mode sanitize, camera/backend init, input shape resolve,
  `--threaded` dispatch.

### 10. Fusion / world model

- Input: `LanePacket`, `ObjectPacket`, pending buffers
- Output: `BehaviorPacket`
- Cần có: drain queues, match by `frame_id`, evict stale, single-mode pass
  through. `BehaviorPacket` hiện là world/fusion snapshot; có thể tách
  `WorldState` sau.

### 11. Safety supervisor / output

- Input: `BehaviorPacket`
- Output: clamped command + rendered/saved/displayed output
- Cần có: clamp steering/throttle, RC hand-off, render, writer, display,
  stop handling. Roadmap: `HealthState` + degraded/safe-stop.

### 12. Multiple-threading architecture

- Input/output qua `LatestQueue`
- Queue hiện tại:
  - `lane_q`: `FramePacket`, maxsize 1
  - `object_q`: `FramePacket`, maxsize 1
  - `lane_state_q`: `LanePacket`, maxsize 2
  - `object_state_q`: `ObjectPacket`, maxsize 2
  - `behavior_q`: `BehaviorPacket`, maxsize 1
- Roadmap: async Hailo inference, health metrics per queue, stale watchdog,
  callback-group style scheduling.

## Recommended Build Order

1. Giữ `ThreadedRuntime` là path chính cho kiến trúc mới.
2. Thêm metrics cho queue drops, per-stage latency, stale evictions.
3. Port validation metrics từ single loop sang threaded runtime.
4. Tách `HealthState` và degraded/safe-stop policy khỏi render/output code.
5. Thử async Hailo backend khi deploy Pi ổn định.
6. Sau khi Python behavior ổn, port từng worker contract sang `cpp_runtime/`.
