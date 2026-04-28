from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
YOLO_DIR = ROOT_DIR / "yolo"
SRC_DIR = ROOT_DIR / "src"


def get_data_yaml():
    return DATA_DIR / "data.yaml"
