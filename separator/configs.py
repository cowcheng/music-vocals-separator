import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO

MDX_C_MODEL_PATH = f"{ROOT_DIR}/separator/weights/MDX23C_D1581.ckpt"
MDX_C_MODEL_CONFIGS_PATH = (
    f"{ROOT_DIR}/separator/configs/mdx_c/model_2_stem_061321.yaml"
)

VR_MODELS_PARAMS_PATH = f"{ROOT_DIR}/separator/configs/vr/4band_v3.json"

VR_DE_ECHO_MODEL_PATH = f"{ROOT_DIR}/separator/weights/UVR-De-Echo-Aggressive.pth"
VR_DE_ECHO_CONFIGS_PATH = f"{ROOT_DIR}/separator/configs/vr/de_echo.yaml"

VR_DE_NOISE_MODEL_PATH = f"{ROOT_DIR}/separator/weights/UVR-DeNoise.pth"
VR_DE_NOISE_CONFIGS_PATH = f"{ROOT_DIR}/separator/configs/vr/de_noise.yaml"
