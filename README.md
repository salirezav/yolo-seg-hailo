# yolo-seg-hailo

Train the pecan instance-segmentation dataset locally, then compile YOLOv8n-seg for the Hailo-8 module in this machine (26 TOPS, firmware 4.23.0, `hailo8`).

The Roboflow export stays in `dataset/` and is not committed. Images are the original resolutions (640×480 through 1920×1200). Training letterboxes to 640, which is the fixed size the Hailo export uses.

## Setup

```powershell
uv sync
```

This environment is Python 3.11 and CUDA 12.1 PyTorch, so both trainers use the RTX 3070. That is also the newest combination with a prebuilt Windows `mmcv` wheel.

## YOLOv8n-seg (Hailo-8 HEF)

For the Hailo AI accelerator, train **YOLOv8n-seg**. Ultralytics Hailo instance-segmentation export supports YOLOv8-seg and YOLO11-seg only; **YOLO26-seg cannot become a HEF**. YOLOv8n-seg is the better Hailo pick (higher INT8 mask retention and faster HEF FPS than YOLO11n-seg). AGPL-3.0.

```powershell
uv run train-yolo-hailo --epochs 100 --patience 30
```

Same command under the explicit name:

```powershell
uv run train-yolov8n-seg --epochs 100 --patience 30
```

That converts COCO polygons and RLE masks into `dataset-yolo/`, then trains `yolov8n-seg.pt` at fixed 640. The run writes `runs/segment/yolov8n-seg-hailo8/weights/best.pt`.

Compile later on a Linux x86_64 machine with Hailo Dataflow Compiler v3.x, then copy the `.hef` to the Pi:

```powershell
yolo export model=runs/segment/yolov8n-seg-hailo8/weights/best.pt format=hailo name=hailo8 imgsz=640 data=dataset-yolo/data.yaml
```

## YOLO26n-seg (non-Hailo)

When you do **not** need a Hailo HEF, train YOLO26n-seg instead. Same Roboflow → `dataset-yolo/` conversion and augment recipe; weights are `yolo26n-seg.pt` (`ultralytics>=8.4.0`). Do not use this checkpoint for `format=hailo` instance-seg export.

```powershell
uv run train-yolo26n-seg --epochs 100 --patience 30
```

The run writes `runs/segment/yolo26n-seg/weights/best.pt`. Deploy with Ultralytics (ONNX, TensorRT, etc.). License is still AGPL-3.0 unless you have an Ultralytics enterprise license.

## RTMDet-Ins-tiny (Apache-2.0)

RTMDet-Ins-tiny is the production model that does not need an Ultralytics enterprise license. Training and deployment go through MMDetection, which is Apache-2.0. This checkpoint is not a Hailo export.

`uv sync` installs this trainer too. Then:

```powershell
uv run train-rtmdet-ins
```

Weights land in `runs/rtmdet-ins-tiny/`. Deploy that checkpoint with MMDetection or MMDeploy.
