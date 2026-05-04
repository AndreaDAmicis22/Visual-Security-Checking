# !pip install ultralytics roboflow python-dotenv onnx
from ultralytics import YOLO

from src.visual_security.utils.paths import DATA_DIR

# -------------------------------------------
# 1️⃣ Define the model
# -------------------------------------------
IMG_SIZE = 800
MODEL_SIZE = "s"  # s, m, l, x
model = YOLO(f"yolo11{MODEL_SIZE}.pt")

# -------------------------------------------
# 2️⃣ Start training
# -------------------------------------------
results = model.train(
    data=str(DATA_DIR / "data.yaml"),
    imgsz=IMG_SIZE,
    device=0,
    epochs=100,
    patience=20,
    batch=16,
    name=f"YOLO_{IMG_SIZE}_{MODEL_SIZE}",
    save_period=5,  # Save every N epochs
    max_det=1000,  # Max detections per image
    workers=4,
    augment=True,
    degrees=0.25,  # random rotation
    translate=0.2,  # random translation
    scale=0.2,  # random scale
    shear=0.0,  # random shear
    perspective=0.0005,  # perspective distortion
    flipud=0.1,  # vertical flip probability
    fliplr=0.4,  # horizontal flip probability
    mosaic=0.4,  # enable mosaic augmentation
    mixup=0.3,  # enable mixup augmentation
    copy_paste=0.1,  # enable copy-paste augmentation
    hsv_h=0.0,  # HSV hue augmentation
    hsv_s=0.2,  # HSV saturation augmentation
    hsv_v=0.1,  # HSV value augmentation
)

# -------------------------------------------
# 3️⃣  Export trained model
# -------------------------------------------
model.export(format="onnx", imgsz=IMG_SIZE, simplify=True, nms=True, opset=18, dynamic=False)

# Export to OpenVINO
# model.export(format="openvino", imgsz=640, optimize=True)
