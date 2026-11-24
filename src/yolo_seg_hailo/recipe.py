"""Fixed training choices for this Hailo-8 module and the Roboflow v3 recipe."""

# hailortcli reports HAILO8 / HM218B1C2FAE. Compile later with DFC v3.x.
HAILO_TARGET = "hailo8"
HAILO_IMGSZ = 640
# Hailo instance-seg HEF: YOLOv8-seg / YOLO11-seg only. YOLO26-seg cannot export.
# YOLOv8n-seg keeps more INT8 mask mAP and runs faster on Hailo than YOLO11n-seg.
YOLO_WEIGHTS = "yolov8n-seg.pt"
# Non-Hailo path: Ultralytics YOLO26 nano segmenter (needs ultralytics>=8.4.0).
YOLO26_WEIGHTS = "yolo26n-seg.pt"

# Roboflow v3 online augmentations. Ultralytics applies hsv_* as random gains.
# Hue of 13 degrees on a 360 degree wheel is 13/360.
YOLO_AUGMENT = {
    "fliplr": 0.5,
    "flipud": 0.5,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "hsv_h": 13 / 360,
    "hsv_s": 0.33,
    "hsv_v": 0.37,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
}

# YOLOXHSVRandomAug takes additive deltas, not Ultralytics gains.
RTMDET_HSV = {
    "hue_delta": 13,
    "saturation_delta": 33,
    "value_delta": 37,
}
