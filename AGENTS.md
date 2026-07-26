## Prompt Routing

When the user gives a normal or incomplete request, do not require a perfect prompt.

First classify the task:
- diagnosis / explanation only
- implementation / code edit
- training / Kaggle / model comparison
- lane segmentation / lane geometry / ADAS behavior
- Hailo / ONNX / preprocessing / model export
- C++ / Pi deploy
- replay / video validation
- research-backed idea discovery
- prompt writing / prompt refinement

Then choose the relevant project instructions or skills.

Default behavior:
- If the request is clear enough, proceed with reasonable assumptions.
- If missing information would make the result risky, ask at most 1-2 focused questions.
- For non-trivial coding tasks, inspect current repo code before editing.
- For diagnosis-only requests, do not edit code.
- For implementation requests, identify root cause first, then make the smallest safe change.
- Always report changed files and validation evidence after edits.
- Do not touch artifacts, datasets, generated outputs, custom configs, or compiled assets unless explicitly asked.

Prompt writing rules for Claude:
- When writing prompts for Claude, keep them concise and task-ready.
  Include only the core goal, relevant context, exact constraints,
  necessary files, validation, and expected report format.
- Do not write long, rambling prompts or restate broad project background
  that is not needed for the current task.
- Do not tell Claude to reread files or docs that it was already asked to
  read in a previous task. Reuse the remembered facts or summarize the
  specific detail needed for the new task.
- Only ask Claude to reopen a previously read file when exact current
  lines, fresh file contents, or edit/validation context is required.

When answering, use this structure when appropriate:
1. What I understood
2. Assumptions
3. Plan
4. Action / patch
5. Validation
6. Remaining risks

For small/simple questions, answer directly and briefly.

## Skill Index

Use `prompt-architect` when the user asks to write, improve, restructure, or translate prompts for Codex, Claude, ChatGPT, or repo-specific workflows.

## Current Session Handoff

Use this section to continue work after the terminal/session is closed.

Project/user context:
- User usually writes Vietnamese. Answer in Vietnamese unless they ask for English prompts.
- Workspace: `D:\adas-v2`.
- Pi project path: `/home/pi/adas-v2`.
- Current target: Raspberry Pi 5 + Hailo ADAS pipeline using STDC1 lane segmentation and YOLO object detection.

Important model/config facts:
- STDC1 lane HEF path: `src/models/stdc1_lane_640.hef`.
- Main STDC1 config: `config/stdc1_lane_hailo.yaml`.
- Do not use `config/full_test.yaml` for STDC1 lane testing; it may still point at older YOLO lane assets.
- Official run command (live CSI camera, real driving use):
  `python src/main.py --config config/stdc1_lane_hailo.yaml --mode full --camera csi --display --threaded`
- Official video-replay command (same shape, --camera swapped for --video):
  `python src/main.py --config config/stdc1_lane_hailo.yaml --mode full --video video_test3.mp4 --display --threaded`
- `--no-display`, `--perf`, and `--fast-overlay` are dev-only diagnostic
  modes. Do not use them as acceptance and do not report headless FPS as
  product success — the operator must see the live UI.

Recent optimization work:
- Threaded lane overlay was refactored to match the single-loop reference by sharing lane finalization/render-corridor logic.
- Expected changed files for that parity work:
  `src/pipeline/lane_finalize.py`, `src/pipeline_loop.py`, `src/pipeline/thread_workers.py`.
- First performance optimization was object detection frame skipping.
- Expected changed files for that optimization:
  `config/stdc1_lane_hailo.yaml`, `src/config_loader.py`, `src/pipeline/threaded_runtime.py`, `src/pipeline/fusion_worker.py`.
- `inference.object_every_n_frames: 3` means object inference should run every third captured frame, while fusion reuses the latest object packet.

Known Pi environment issue:
- If Python crashes before project code runs and `python -X faulthandler -c "import cv2"` segfaults, the issue is the Pi OpenCV/native environment, not the ADAS code.
- Previously observed: `import ssl` worked, but `import cv2` segfaulted.

Useful sync commands from Windows dev to Pi:
- Sync main code/config:
  `scp -r src config scripts pi@raspi:/home/pi/adas-v2/`
- Sync only threaded parity files:
  `scp src/pipeline/lane_finalize.py pi@raspi:/home/pi/adas-v2/src/pipeline/`
  `scp src/pipeline_loop.py pi@raspi:/home/pi/adas-v2/src/`
  `scp src/pipeline/thread_workers.py pi@raspi:/home/pi/adas-v2/src/pipeline/`
- Sync object-skip optimization files:
  `scp config/stdc1_lane_hailo.yaml pi@raspi:/home/pi/adas-v2/config/`
  `scp src/config_loader.py pi@raspi:/home/pi/adas-v2/src/`
  `scp src/pipeline/threaded_runtime.py src/pipeline/fusion_worker.py pi@raspi:/home/pi/adas-v2/src/pipeline/`

Dev-machine validation:
- Use `--onnx` on the Windows dev machine because Hailo hardware is only on the Pi.
- Syntax check:
  `.\.venv\Scripts\python.exe -m py_compile src\config_loader.py src\pipeline\threaded_runtime.py src\pipeline\fusion_worker.py src\pipeline_loop.py`
