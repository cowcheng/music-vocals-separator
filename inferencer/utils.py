import io
import logging
import math
from os import sendfile
import traceback

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

    Parameters:
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


def write_array_to_mem(audio_data, subtype):
    """
    Writes audio data to a memory buffer in WAV format.
    If `audio_data` is a NumPy array, it is written to an in-memory WAV file with a sample rate of 44100 Hz
    using the specified subtype. The resulting buffer is returned. If `audio_data` is not a NumPy array,
    it is returned unchanged.

    Parameters:
        audio_data (np.ndarray or any): The audio data to be written. If a NumPy array, it will be converted to WAV format.
        subtype (str): The subtype for the WAV file format.

    Returns:
        io.BytesIO or any: A BytesIO buffer containing the WAV data if `audio_data` is a NumPy array, otherwise the original `audio_data`.
    """

    if isinstance(audio_data, np.ndarray):
        audio_buffer = io.BytesIO()
        sendfile.write(audio_buffer, audio_data, 44100, subtype=subtype, format="WAV")
        audio_buffer.seek(0)
        return audio_buffer
    else:
        return audio_data


def wave_to_spectrogram(wave, hop_length, n_fft, mp, band, is_v51_model=False):
    """
    Converts a waveform into a spectrogram.
    This function processes a stereo or mono waveform and transforms it into a spectrogram using the Short-Time Fourier Transform (STFT). Depending on the model version and parameters provided, it can perform operations such as reversing the waveform, mid-side processing, or channel conversion.

    Parameters:
        wave (np.ndarray): Input waveform array. Can be mono or stereo.
        hop_length (int): Number of samples between successive frames in the STFT.
        n_fft (int): Length of the FFT window.
        mp (module or object): Parameter object containing configuration settings.
        band (int): Band parameter used for channel conversion in specific models.
        is_v51_model (bool, optional): Flag to indicate if the V51 model is used. Defaults to False.

    Returns:
        np.ndarray: Spectrogram representation of the input waveform.
    """

    if wave.ndim == 1:
        wave = np.asfortranarray([wave, wave])

    if not is_v51_model:
        if mp.param["reverse"]:
            wave_left = np.flip(np.asfortranarray(wave[0]))
            wave_right = np.flip(np.asfortranarray(wave[1]))
        elif mp.param["mid_side"]:
            wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
        elif mp.param["mid_side_b2"]:
            wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
        else:
            wave_left = np.asfortranarray(wave[0])
            wave_right = np.asfortranarray(wave[1])
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])

    if is_v51_model:
        spec = convert_channels(spec, mp, band)

    return spec


def convert_channels(spec, mp, band):
    """
    Convert audio channels based on the specified conversion mode.

    Parameters:
        spec (numpy.ndarray): Input spectral data representing stereo channels, usually with shape [2, ...].
        mp (object): Parameter object containing configuration settings. It should have a 'param' attribute structured as mp.param["band"][band].
        band (str): Identifier for the specific band to retrieve the 'convert_channels' setting.

    Returns:
        numpy.ndarray: The converted spectral data as a Fortran-contiguous array with the adjusted left and right channels.
        If the 'convert_channels' mode is unrecognized, returns the original 'spec' array.
    """

    cc = mp.param["band"][band].get("convert_channels")

    if "mid_side_c" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25)
        spec_right = np.subtract(spec[1], spec[0] * 0.25)
    elif "mid_side" == cc:
        spec_left = np.add(spec[0], spec[1]) / 2
        spec_right = np.subtract(spec[0], spec[1])
    elif "stereo_n" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25) / 0.9375
        spec_right = np.add(spec[1], spec[0] * 0.25) / 0.9375
    else:
        return spec

    return np.asfortranarray([spec_left, spec_right])


def combine_spectrograms(specs, mp, is_v51_model=False):
    """
    Combines multiple spectrograms into a single complex spectrogram.

    Parameters:
        specs (dict): A dictionary of spectrograms to be combined. Each key corresponds to a band index,
                      and each value is a NumPy array representing the spectrogram for that band.
        mp (object): An object containing parameters required for combination, accessible via the `param` attribute.
                     Expected `param` keys include:
                         - "bins" (int): Total number of frequency bins.
                         - "band" (dict): Definitions of frequency bands with "crop_start" and "crop_stop".
                         - "pre_filter_start" (int): Starting frequency bin for pre-filtering.
                         - "pre_filter_stop" (int): Stopping frequency bin for pre-filtering.
        is_v51_model (bool, optional): Flag indicating whether to apply the pre-filter using the v51 model.
                                       Defaults to False.

    Returns:
        np.ndarray: A combined complex spectrogram with shape (2, bins + 1, l), where `bins` is defined in `mp.param["bins"]`
                    and `l` is the minimum length among the provided spectrograms.

    Raises:
        ValueError: If the total number of bins after combination exceeds `mp.param["bins"]`.
    """

    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][
            :, mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"], :l
        ]
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    # lowpass fiter

    if mp.param["pre_filter_start"] > 0:
        if is_v51_model:
            spec_c *= get_lp_filter_mask(
                spec_c.shape[1],
                mp.param["pre_filter_start"],
                mp.param["pre_filter_stop"],
            )
        else:
            if bands_n == 1:
                spec_c = fft_lp_filter(
                    spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"]
                )
            else:
                gp = 1
                for b in range(
                    mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]
                ):
                    g = math.pow(
                        10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0
                    )
                    gp = g
                    spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)


def get_lp_filter_mask(n_bins, bin_start, bin_stop):
    """
    Generates a low-pass filter mask for spectral processing.

    Parameters:
        n_bins (int): Total number of frequency bins.
        bin_start (int): The bin index where the filter starts to attenuate.
        bin_stop (int): The bin index where the filter ends attenuation.

    Returns:
        np.ndarray: A mask array of shape (n_bins, 1) with ones up to bin_start - 1,
                    linearly decreasing from bin_start to bin_stop,
                    and zeros thereafter.
    """

    mask = np.concatenate(
        [
            np.ones((bin_start - 1, 1)),
            np.linspace(1, 0, bin_stop - bin_start + 1)[:, None],
            np.zeros((n_bins - bin_stop, 1)),
        ],
        axis=0,
    )

    return mask


def fft_lp_filter(spec, bin_start, bin_stop):
    """
    Applies a linear low-pass filter to the spectrogram `spec` between the specified frequency bins.
    This function attenuates the frequency components from `bin_start` to `bin_stop` linearly and zeroes out frequencies above `bin_stop`.

    Parameters:
        spec (numpy.ndarray): The spectrogram to be filtered, with shape (batch_size, frequency_bins, time_frames).
        bin_start (int): The starting frequency bin for applying the low-pass filter.
        bin_stop (int): The stopping frequency bin where the filter ends.

    Returns:
        numpy.ndarray: The filtered spectrogram with the same shape as the input.
    """

    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0

    return spec


def preprocess(X_spec):
    """
    Separates the magnitude and phase components of the input spectrogram.

    Parameters:
        X_spec (numpy.ndarray): Complex spectrogram array with shape (frequency_bins, time_frames).

    Returns:
        tuple:
            X_mag (numpy.ndarray): Magnitude of the spectrogram.
            X_phase (numpy.ndarray): Phase of the spectrogram.
    """

    X_mag = np.abs(X_spec)
    X_phase = np.angle(X_spec)

    return X_mag, X_phase


def make_padding(width, cropsize, offset):
    """
    Generates padding values based on the specified width, crop size, and offset.

    Parameters:
        width (int): The total width for which padding is to be calculated.
        cropsize (int): The size of the crop area.
        offset (int): The offset value used to determine the left padding.

    Returns:
        tuple: A tuple containing left padding, right padding, and the region of interest size.
    """

    left = offset
    roi_size = cropsize - offset * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def adjust_aggr(mask, is_non_accom_stem, aggressiveness):
    """
    Adjust the aggregation mask based on aggressiveness settings.
    This function modifies the input mask by applying power transformations
    to different sections of the mask based on the provided aggressiveness parameters.
    It accounts for whether the stem is non-accompaniment and applies corrections
    to the left and right channels if specified.

    Parameters:
        mask (numpy.ndarray): The mask to be adjusted with shape (2, ...).
        is_non_accom_stem (bool): Flag indicating if the stem is non-accompaniment.
        aggressiveness (dict): Dictionary containing aggressiveness settings.
            - value (float): Base aggressiveness value.
            - aggr_correction (dict or None): Optional corrections for left and right channels.
                - left (float): Correction value for the left channel.
                - right (float): Correction value for the right channel.
            - split_bin (int): The bin index at which to split the mask for applying different adjustments.

    Returns:
        numpy.ndarray: The adjusted mask after applying aggressiveness transformations.
    """

    aggr = aggressiveness["value"] * 2

    if aggr != 0:
        if is_non_accom_stem:
            aggr = 1 - aggr

        aggr = [aggr, aggr]

        if aggressiveness["aggr_correction"] is not None:
            aggr[0] += aggressiveness["aggr_correction"]["left"]
            aggr[1] += aggressiveness["aggr_correction"]["right"]

        for ch in range(2):
            mask[ch, : aggressiveness["split_bin"]] = np.power(
                mask[ch, : aggressiveness["split_bin"]], 1 + aggr[ch] / 3
            )
            mask[ch, aggressiveness["split_bin"] :] = np.power(
                mask[ch, aggressiveness["split_bin"] :], 1 + aggr[ch]
            )

    return mask


def merge_artifacts(y_mask, thres=0.01, min_range=64, fade_size=32):
    """
    Merge artifacts in the given mask based on threshold, minimum range, and fade size.
    This function processes a binary mask to merge artifact regions that exceed a specified
    minimum range. It applies a fading effect at the boundaries to smooth transitions.

    Parameters:
        y_mask (numpy.ndarray): The input mask with shape (channels, height, width).
        thres (float, optional): Threshold value to determine artifact regions. Defaults to 0.01.
        min_range (int, optional): Minimum range of artifact to be merged. Must be >= fade_size * 2. Defaults to 64.
        fade_size (int, optional): Size of the fade effect applied at the boundaries. Defaults to 32.

    Returns:
        numpy.ndarray: The processed mask with merged artifacts.

    Raises:
        ValueError: If `min_range` is less than `fade_size * 2`.
    """

    mask = y_mask

    try:
        if min_range < fade_size * 2:
            raise ValueError("min_range must be >= fade_size * 2")

        idx = np.where(y_mask.min(axis=(0, 1)) > thres)[0]
        start_idx = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        end_idx = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        artifact_idx = np.where(end_idx - start_idx > min_range)[0]
        weight = np.zeros_like(y_mask)
        if len(artifact_idx) > 0:
            start_idx = start_idx[artifact_idx]
            end_idx = end_idx[artifact_idx]
            old_e = None
            for s, e in zip(start_idx, end_idx):
                if old_e is not None and s - old_e < fade_size:
                    s = old_e - fade_size * 2

                if s != 0:
                    weight[:, :, s : s + fade_size] = np.linspace(0, 1, fade_size)
                else:
                    s -= fade_size

                if e != y_mask.shape[2]:
                    weight[:, :, e - fade_size : e] = np.linspace(1, 0, fade_size)
                else:
                    e += fade_size

                weight[:, :, s + fade_size : e - fade_size] = 1
                old_e = e

        v_mask = 1 - y_mask
        y_mask += weight * v_mask

        mask = y_mask
    except Exception as e:
        error_name = f"{type(e).__name__}"
        traceback_text = "".join(traceback.format_tb(e.__traceback__))
        message = f'{error_name}: "{e}"\n{traceback_text}"'
        print("Post Process Failed: ", message)

    return mask


def cmb_spectrogram_to_wave(
    spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False
):
    """
    Converts a combined spectrogram into a waveform.
    This function processes a combined spectrogram (`spec_m`) by applying various filters and transformations
    based on the provided model parameters (`mp`). It handles multiple frequency bands, applies high-pass
    and low-pass filters as needed, and resamples the resulting waveform to match the target sample rates.

    Parameters:
        spec_m (numpy.ndarray):
            The input spectrogram with shape (channels, frequency_bins, time_frames).
        mp (object):
            The model parameters containing band configurations and other settings.
        extra_bins_h (int, optional):
            The number of extra high-frequency bins to process. Defaults to None.
        extra_bins (numpy.ndarray, optional):
            Additional frequency bins to include in the processing. Defaults to None.
        is_v51_model (bool, optional):
            Flag indicating whether to use version 5.1 model filters. Defaults to False.

    Returns:
        numpy.ndarray:
            The reconstructed waveform as a 1D NumPy array.
    """

    bands_n = len(mp.param["band"])
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.ndarray(
            shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex
        )
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[
            :, offset : offset + h, :
        ]

        offset += h
        if d == bands_n:  # higher
            if extra_bins_h:  # if --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[
                    :, :extra_bins_h, :
                ]
            if bp["hpf_start"] > 0:
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(
                        spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1
                    )
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
            else:
                wave = np.add(
                    wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
                )
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lower
                if is_v51_model:
                    spec_s *= get_lp_filter_mask(
                        spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"]
                    )
                else:
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = librosa.resample(
                    spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model),
                    bp["sr"],
                    sr,
                    res_type="sinc_fastest",
                )
            else:  # mid
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(
                        spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1
                    )
                    spec_s *= get_lp_filter_mask(
                        spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"]
                    )
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                wave2 = np.add(
                    wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
                )
                wave = librosa.resample(wave2, bp["sr"], sr, res_type="sinc_fastest")

    return wave


def get_hp_filter_mask(n_bins, bin_start, bin_stop):
    """
    Generate a high-pass filter mask.

    Parameters:
        n_bins (int): Total number of frequency bins.
        bin_start (int): The starting bin index where the mask begins to increase.
        bin_stop (int): The stopping bin index where the mask completes its transition.

    Returns:
        numpy.ndarray: A mask array of shape (n_bins, 1) where values transition from 0 to 1
                       between `bin_stop` and `bin_start`, are 0 below `bin_stop`, and 1 above `bin_start`.
    """

    mask = np.concatenate(
        [
            np.zeros((bin_stop + 1, 1)),
            np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None],
            np.ones((n_bins - bin_start - 2, 1)),
        ],
        axis=0,
    )

    return mask


def fft_hp_filter(spec, bin_start, bin_stop):
    """
    Applies a high-pass filter to the given frequency spectrum.
    This function iteratively reduces the magnitude of the spectrum within a specified
    range of frequency bins and zeroes out the lower frequency bins up to `bin_stop`.

    Parameters:
        spec (numpy.ndarray): The input spectrum to be filtered, with shape (batch_size, num_bins, num_channels).
        bin_start (int): The starting index of the frequency bin range to apply the high-pass filter.
        bin_stop (int): The ending index of the frequency bin range to apply the high-pass filter.

    Returns:
        numpy.ndarray: The filtered spectrum with the high-pass filter applied.
    """

    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0

    return spec


def spectrogram_to_wave(spec, hop_length=1024, mp={}, band=0, is_v51_model=True):
    """
    Converts a spectrogram to a stereo wave signal.

    Parameters:
        spec (np.ndarray): Input spectrogram with shape (2, n_freq_bins, n_time_steps),
            where spec[0] and spec[1] correspond to the left and right channels respectively.
        hop_length (int, optional): Number of samples between successive frames for the inverse STFT. Defaults to 1024.
        mp (dict, optional): Parameter dictionary containing model configurations such as channel conversion settings. Defaults to an empty dictionary.
        band (int, optional): Frequency band index used for channel conversion when `is_v51_model` is True. Defaults to 0.
        is_v51_model (bool, optional): Flag indicating whether to use the v5.1 model's channel conversion logic. Defaults to True.

    Returns:
        np.ndarray: A NumPy array in Fortran order containing the left and right waveform signals after conversion.
    """

    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if is_v51_model:
        cc = mp.param["band"][band].get("convert_channels")
        if "mid_side_c" == cc:
            return np.asfortranarray(
                [
                    np.subtract(wave_left / 1.0625, wave_right / 4.25),
                    np.add(wave_right / 1.0625, wave_left / 4.25),
                ]
            )
        elif "mid_side" == cc:
            return np.asfortranarray(
                [
                    np.add(wave_left, wave_right / 2),
                    np.subtract(wave_left, wave_right / 2),
                ]
            )
        elif "stereo_n" == cc:
            return np.asfortranarray(
                [
                    np.subtract(wave_left, wave_right * 0.25),
                    np.subtract(wave_right, wave_left * 0.25),
                ]
            )
    else:
        if mp.param["reverse"]:
            return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
        elif mp.param["mid_side"]:
            return np.asfortranarray(
                [
                    np.add(wave_left, wave_right / 2),
                    np.subtract(wave_left, wave_right / 2),
                ]
            )
        elif mp.param["mid_side_b2"]:
            return np.asfortranarray(
                [
                    np.add(wave_right / 1.25, 0.4 * wave_left),
                    np.subtract(wave_left / 1.25, 0.4 * wave_right),
                ]
            )

    return np.asfortranarray([wave_left, wave_right])
