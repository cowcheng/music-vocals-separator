import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO

MDX_C_MODEL_PATH = f"{ROOT_DIR}/inferencer/weights/MDX23C_D1581.ckpt"
MDX_C_MODEL_CONFIGS_PATH = (
    f"{ROOT_DIR}/inferencer/configs/mdx_c/model_2_stem_061321.yaml"
)

VR_DE_ECHO_MODEL_PATH = (
    f"{ROOT_DIR}/inferencer/weights/UVR-De-Echo-Aggressive.pth"
)
VR_DE_ECHO_CONFIGS_PATH = f"{ROOT_DIR}/inferencer/configs/vr/de_echo.yaml"
VR_DE_ECHO_MODELS_PARAMS_PATH = f"{ROOT_DIR}/inferencer/configs/vr/4band_v3.json"
