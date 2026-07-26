# CLAUDE.md

This repository is an ADAS stack for Raspberry Pi 5 + Hailo-8. The real ship
path is the Python runtime on Pi, with ONNX fallback for Windows development:

1. build and stabilize behavior in Python
2. validate it with replay tooling and focused static checks
3. deploy the same Python path to Raspberry Pi with Hailo HEF models

Optimize for that workflow, not generic demos or framework rewrites.

## Working Contract

- Prefer repository code over generic examples.
- For non-trivial coding, debugging, or research tasks, apply the
  `engineering_research` skill first.
- When code, scripts, skills, and docs disagree, use this priority:
  1. `src/`
  2. `scripts/`
  3. this file and `.claude/skills/`
  4. older notes under `docs/`
- Keep edits small and task-shaped. Do not turn the repo into a generic
  perception framework.
- Keep context task-scoped. Do not read files, directories, generated
  outputs, or artifacts that are not relevant to the current task. Once a
  file or document has been read during the current workflow, rely on the
  remembered facts or a short local note instead of re-reading it repeatedly;
  only reopen it when the exact line/content is needed for an edit or
  verification.
- Use the relevant skill for specialized work (see *Skill Index* below).

## Project Truths (always-on)

- Two runtime/model contexts are tracked in this repository. Do not collapse
  them into one naming scheme:
  - **Python reference / newer development path:** **`YOLO26s_seg`** for lane
    segmentation and **`YOLO26s`** for object detection.
  - **Thesis / C++ Hailo runtime path:** **`STDC1`** lane semantic
    segmentation via `src/models/stdc1_lane_640.hef` and **`YOLO26n`** object
    detection via `src/models/yolo26n_object.hef`.
- The graduation slide deck and `cpp/README.md` describe the thesis/C++ Hailo
  path. Do not "correct" those references to `YOLO26s_seg` unless the user
  explicitly asks to retarget the presentation to the Python reference path.
- DeepLabV3+MobileNetV2 is **not used**. Do not add, restore, compile,
  configure, or document DeepLab paths unless the user explicitly asks for
  historical cleanup.
- Runtime modes are `lane`, `detection`, `full`. Lane and detection are
  independent branches; neither should silently become mandatory for the
  other.
- **Hailo inputs are RGB `uint8`. Do not divide by 255 on-device.**
- ONNX fallback normalizes to `float32 / 255.0` inside
  `src/pipeline/onnx_backend.py`.
- `config/*_custom.yaml` files are **user-local overrides** — do not edit
  unless the user explicitly asks.
- `src/models/*.hef`, `src/models/*.onnx`, `checkpoints/`, `calib_data/`,
  and `calib_images/` are artifacts or datasets — do not hand-edit unless
  the task is explicitly artifact / data prep.
- For Python reference Pi/Hailo runs, the matching `yolo26s_seg.hef` is
  present. For thesis/C++ Hailo runs, use `stdc1_lane_640.hef` and
  `yolo26n_object.hef`. ONNX fallback (`--onnx`) auto-swaps `.hef` paths to
  `.onnx` and is the working dev path on Windows.
- **Live snapshot:** `PROJECT_STATE.md` at the repo root tracks the
  current runtime path, the active lane checkpoint, recent outputs, and
  recent validation status. Read it before assuming what the working
  config or model is on a given day.

## Architecture (big picture)

`src/main.py` is a thin orchestrator. Per run it loads YAML config →
applies CLI overrides (`config_loader.apply_cli_overrides`) → applies
the runtime mode (`runtime_mode.apply_runtime_mode`, then
`sanitize_runtime_features`) → opens `CameraSource` → constructs
inference via `pipeline.inference.create_inference` (Hailo if
`/dev/hailo0` exists and `--onnx` is not set, else
`OnnxFallbackInference`). It then dispatches to one of three loop
strategies:

- **Single-threaded** (default): `pipeline_loop.run_pipeline_frame`
  drives preprocess → inference → lane branch + object branch
  (optionally parallelized on a `ThreadPoolExecutor` when
  `pipeline.branch_threading=true` and both branches are enabled) →
  fusion → behavior → controller → render/RC, all on the main loop.
- **Threaded** (`--threaded`): `pipeline.threaded_runtime.run_threaded`
  runs `Capture → lane_q + object_q → LaneWorker / ObjectWorker →
  FusionWorker → output thread`. Each stage uses
  `LatestQueue(maxsize=1)` and drops stale frames; fusion matches lane
  and object packets by `frame_id` with `STALE_FRAME_TOLERANCE` slack.

Package boundaries:

- `src/pipeline/` — I/O and infra: camera source, Hailo + ONNX
  backends, pre/post-processors, threaded runtime + workers, packets,
  latest-queue, output renderer, overlays.
- `src/features/` — perception and control logic: lane decode + the
  `lane_boundary_tracker_*` family, dynamic corridor, red-center
  geometry, YOLO26 decode + tracker + localizer, behavior + controller,
  ego-speed, validation metrics.
- `src/pipeline_loop.py` owns module-level stateful singletons
  (`_LANE_BOUNDARY_TRACKER`, `_OBJECT_TRACKER`, `_DYNAMIC_CORRIDOR`,
  `_EGO_SPEED_ESTIMATOR`). Use the `reset_*` helpers when starting a
  fresh video so state does not leak across runs.

Model artifact resolution: `config_loader._resolve_lane_segmentation_hef`
and `_resolve_yolo26_hef` walk candidate lists, and
`OnnxFallbackInference` separately resolves ONNX candidates under
`src/models/` and `checkpoints/`. A user-supplied `--lane-onnx PATH`
overrides the lane model path.

## Repo Layout

- `src/` — Python runtime (orchestrator, pipeline, features). See
  *Architecture* above.
- `config/` — YAML configs. `base.yaml` is the shared baseline;
  `full_test.yaml` / `lane_only.yaml` / `detection_only.yaml` map to the
  three runtime modes; `config/hailo/` holds Hailo compile YAMLs;
  `*_custom.yaml` are user-local overrides (do not edit).
- `scripts/` — operator scripts: Hailo compile (`compile_hailo_*.sh`),
  Pi setup (`install.sh`, `pi_prepare_runtime.sh`),
  ONNX export (`export_yolo26*.py`), dataset prep, safety checks.
- `tools/` — visual debug / preview utilities for lane geometry,
  corridor comparison, frame sampling. Read-only inspection helpers,
  not part of the runtime.
- `checkpoints/`, `src/models/`, `calib_data/`, `calib_images/` —
  artifacts and datasets. Do not hand-edit unless task is artifact/data
  prep.
- `outputs/`, `logs/`, `tmp*` — generated outputs and scratch.
- `docs/` — older notes (lower priority than `src/` and skills per the
  *Working Contract*).
  agents. Overlaps with this file; both should stay consistent.
- `PROJECT_STATE.md` — live snapshot (re-read each session).

## Behavioral Guardrails

- **Think before coding.** State assumptions; surface tradeoffs; push back
  on requests that revive obsolete paths or weaken validation.
- **Simplicity first.** Write the minimum that solves the requested
  behavior. No speculative features, generic frameworks, or config knobs.
- **Surgical changes.** Touch only files needed for the task. Do not
  refactor, rename, or reformat unrelated code. If unrelated dead code is
  noticed, mention it in the report — do not delete it.
- **Goal-driven.** Define success criteria before editing; reproduce
  failures when practical; validate with the narrowest useful replay or
  static check and report concrete evidence.
- **No guessing.** Inspect the current code, contract, or runtime mode
  when unclear. If it remains unclear and implementation would be risky,
  ask or stop.

## Editing Rules

- Keep Python files under roughly **200 lines** when practical. Split by
  responsibility past that.
- Reuse existing helpers before adding wrappers.
- Every changed line must trace to the user request or to validation
  needed for that request.
- Diagnosis-only requests → do not edit code.
- Implementation requests → minimal patch + validation evidence.
- The `tests/` suite has been removed; use replay commands, CLI checks,
  targeted scripts, or static checks for validation unless tests are
  explicitly reintroduced.
- Do not change artifacts, datasets, generated outputs, local configs, or
  compiled assets unless explicitly requested.
- No speculative config flags.

## Skill Index

Use the matching skill for specialized work — load it, follow its rules:

| Topic                                                            | Skill                  |
|------------------------------------------------------------------|------------------------|
| Lane segmentation, `lane_area`, purple boundaries, red center, yellow corridor, lane controller, `pipeline_loop` lane branch | `yolo26s_seg`          |
| Object detection, YOLO26 decode, class mapping, behavior inputs   | `yolo26`               |
| Hailo backend, HEF loading, compile flow, Pi runtime I/O          | `hailo`                |
| ONNX fallback, x86 dev path, normalization, shape handling        | `onnx`                 |
| Pi packaging, deploy scripts, runtime payload                     | `deploy`               |
| Camera / capture / runtime modes / threaded pipeline topology     | `gstreamer`            |
| Replay validation, run commands, runtime flags, FPS evidence      | `replay-validation`    |
| Kaggle / Ultralytics training, resume vs fine-tune, dataset audit | `yolo26-training-kaggle` |
| Research-backed ideas from papers, academic references, algorithm inspiration, method comparison | `research-ideas-from-papers` |
| Default investigation / verification loop for any non-trivial task| `engineering_research` |

If a task spans skills (e.g. lane decode + Hailo compile), load both.

## Quick Commands

```powershell
# Primary replay (full mode, ONNX fallback, with display)
```
