import os
import shutil

from dotenv import load_dotenv
from roboflow import Roboflow

from src.visual_security.utils.paths import DATA_DIR, ROOT_DIR

load_dotenv(ROOT_DIR / ".env")


def download():
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("akfa-beqxl").project("safety-rd-v1")
    version = project.version(1)

    # Scarica
    raw_dataset = version.download("yolov11")

    # Rinomina e sposta in /data nella root
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    os.rename(raw_dataset.location, DATA_DIR)
    print(f"Dataset pronto in {DATA_DIR}")


if __name__ == "__main__":
    download()
