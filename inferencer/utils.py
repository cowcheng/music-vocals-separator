import logging

import librosa
import numpy as np
import yaml
from ml_collections import ConfigDict

from inferencer.configs import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=LOG_LEVEL,
)

logger = logging.getLogger(name="MusicVocalsSeparator")

"""
Source from the UVR5 repository
"""


def load_model_configs_file(config_path: str):
    """
    Loads model configurations from a YAML file.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        ConfigDict: The loaded configuration as a ConfigDict object.
    """

    with open(config_path) as f:
        yaml_file = yaml.load(
            stream=f,
            Loader=yaml.FullLoader,
        )
        config = ConfigDict(yaml_file)
        return config


def load_audio_file(audio_path: str):
    """
    Load an audio file.

    Parameters:
        audio_path (str): Path to the audio file.

    Returns:
        np.ndarray: Loaded audio data as a NumPy array with shape (channels, samples).
    """

    audio_npy, _ = librosa.load(
        audio_path,
        mono=False,
        sr=44100,
    )
    if audio_npy.ndim == 1:
        audio_npy = np.asfortranarray([audio_npy, audio_npy])
    return audio_npy


def change_pitch_semitones(y, sr, semitone_shift):
    """
    Adjusts the pitch of audio signals by a specified number of semitones.

    Parameters:
        y (np.ndarray): Input audio signal(s), shape (n_channels, n_samples).
        sr (int): Original sampling rate of the audio signal.
        semitone_shift (float): Number of semitones to shift the pitch. Positive values shift up, negative values shift down.

    Returns:
        Tuple[np.ndarray, float]:
            - y_pitch_tuned (np.ndarray): Pitch-shifted audio signal(s).
            - new_sr (float): New sampling rate after pitch shifting.
    """

    factor = 2 ** (semitone_shift / 12)
    y_pitch_tuned = []
    for y_channel in y:
        y_pitch_tuned.append(
            librosa.resample(y_channel, sr, sr * factor, res_type="sinc_fastest")
        )
    y_pitch_tuned = np.array(y_pitch_tuned)
    new_sr = sr * factor
    return y_pitch_tuned, new_sr


def match_array_shapes(array_1: np.ndarray, array_2: np.ndarray, is_swap=False):
    """
    Adjusts the shape of `array_1` to match that of `array_2` by truncating or padding as necessary.
    If `is_swap` is `True`, both arrays are transposed before and after the shape adjustment.

    Parameters:
        array_1 (np.ndarray): The first array to be adjusted.
        array_2 (np.ndarray): The second array whose shape is used as the target.
        is_swap (bool, optional): Determines whether to transpose the arrays before matching. Defaults to `False`.

    Returns:
        np.ndarray: The adjusted `array_1` with the same shape as `array_2`.
    """

    if is_swap:
        array_1, array_2 = array_1.T, array_2.T
    if array_1.shape[1] > array_2.shape[1]:
        array_1 = array_1[:, : array_2.shape[1]]
    elif array_1.shape[1] < array_2.shape[1]:
        padding = array_2.shape[1] - array_1.shape[1]
        array_1 = np.pad(array_1, ((0, 0), (0, padding)), "constant", constant_values=0)
    if is_swap:
        array_1, array_2 = array_1.T, array_2.T
    return array_1
