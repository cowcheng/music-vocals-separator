import math
import os
from typing import List

import librosa
import numpy as np
import soundfile
import torch
from tqdm import tqdm

from inferencer import utils
from inferencer.configs import (
    VR_DE_NOISE_CONFIGS_PATH,
    VR_DE_NOISE_MODEL_PATH,
    VR_MODELS_PARAMS_PATH,
)
from inferencer.models.vr.cascaded_net import CascadedNet
from inferencer.models.vr.vr_model_param import ModelParameters


class VRDeNoiseInferencer:
    def __init__(
        self,
        output_dir: str,
        device: str,
    ) -> None:
        """
        Initialize the VRDeNoiseInferencer.
        
        Args:
            output_dir (str): Directory where output files will be saved.
            device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        
        Attributes:
            output_dir (str): Directory for saving output files.
            device (torch.device): Torch device used for model computations.
            model_data (dict): Configuration parameters loaded for the model.
            model_params (ModelParameters): Parameters specific to the model.
            model (CascadedNet): The neural network model used for denoising.
            is_vr_51_model (bool): Indicates if the VR 5.1 model is used.
            sample_rate (int): The sample rate for audio processing.
            wav_type_set (str): WAV file format type.
            window_size (int): Size of the processing window.
            batch_size (int): Number of samples processed in a batch.
            aggressiveness (dict): Settings controlling the denoising aggressiveness.
        """

        self.output_dir = output_dir
        self.output_dir = output_dir
        self.device = torch.device(
            device=device if torch.cuda.is_available() else "cpu"
        )
        self.model_data = utils.load_model_configs_file(
            config_path=VR_DE_NOISE_CONFIGS_PATH
        )["params"]
        self.model_params = ModelParameters(config_path=VR_MODELS_PARAMS_PATH)
        nn_arch_sizes = [
            31191,  # default
            33966,
            56817,
            123821,
            123812,
            129605,
            218409,
            537238,
            537227,
        ]
        model_size = math.ceil(os.stat(VR_DE_NOISE_MODEL_PATH).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key=lambda x: abs(x - model_size))
        self.model = CascadedNet(
            self.model_params.param["bins"] * 2,
            nn_arch_size,
            nout=self.model_data["nout"],
            nout_lstm=self.model_data["nout_lstm"],
        )
        self.is_vr_51_model = True
        self.model.load_state_dict(
            state_dict=torch.load(
                f=VR_DE_NOISE_MODEL_PATH,
                map_location=torch.device(device="cpu"),
            )
        )
        self.model.to(device)
        self.sample_rate = 44100
        self.wav_type_set = "PCM_16"
        self.window_size = 512
        self.batch_size = 1
        self.aggressiveness = {
            "value": 0.05,
            "split_bin": self.model_params.param["band"][1]["crop_stop"],
            "aggr_correction": None,
        }

    def infer(
        self,
        files_list: List[str],
    ) -> None:
        """
        Perform inference to denoise vocal tracks from a list of audio files.
        
        This method processes each audio file in the provided `files_list` by:
        - Loading and resampling the audio waveform across multiple frequency bands.
        - Generating spectrograms for each band.
        - Combining spectrograms and preprocessing for model input.
        - Applying the trained model to predict masks for noise reduction.
        - Adjusting and applying the mask to obtain the denoised spectrogram.
        - Converting the processed spectrogram back to waveform and saving the output audio.
        
        Args:
            files_list (List[str]): A list of file paths to the audio files to be processed.
        
        Raises:
            Exception: If the predicted mask dimensions are incompatible or if no mask is generated.
        """
        for file in tqdm(
            iterable=files_list,
            desc=f"VR DE NOISE inferencing on device {self.device}",
        ):
            filename = file.split("/")[-1].split(".")[0]
            X_wave, X_spec_s = {}, {}
            bands_n = len(self.model_params.param["band"])
            audio_file = utils.write_array_to_mem(file, subtype=self.wav_type_set)
            for d in range(bands_n, 0, -1):
                bp = self.model_params.param["band"][d]
                wav_resolution = bp["res_type"]
                if d == bands_n:
                    X_wave[d], _ = librosa.load(
                        audio_file,
                        bp["sr"],
                        False,
                        dtype=np.float32,
                        res_type=wav_resolution,
                    )
                    X_spec_s[d] = utils.wave_to_spectrogram(
                        X_wave[d],
                        bp["hl"],
                        bp["n_fft"],
                        self.model_params,
                        band=d,
                        is_v51_model=self.is_vr_51_model,
                    )
                    if X_wave[d].ndim == 1:
                        X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
                else:
                    X_wave[d] = librosa.resample(
                        X_wave[d + 1],
                        self.model_params.param["band"][d + 1]["sr"],
                        bp["sr"],
                        res_type=wav_resolution,
                    )
                    X_spec_s[d] = utils.wave_to_spectrogram(
                        X_wave[d],
                        bp["hl"],
                        bp["n_fft"],
                        self.model_params,
                        band=d,
                        is_v51_model=self.is_vr_51_model,
                    )
                if d == bands_n:
                    self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                        self.model_params.param["pre_filter_stop"]
                        - self.model_params.param["pre_filter_start"]
                    )
                    self.input_high_end = X_spec_s[d][
                        :,
                        bp["n_fft"] // 2 - self.input_high_end_h : bp["n_fft"] // 2,
                        :,
                    ]
            X_spec = utils.combine_spectrograms(
                X_spec_s, self.model_params, is_v51_model=self.is_vr_51_model
            )
            del X_wave, X_spec_s, audio_file
            X_mag, X_phase = utils.preprocess(X_spec)
            n_frame = X_mag.shape[2]
            pad_l, pad_r, roi_size = utils.make_padding(
                n_frame, self.window_size, self.model.offset
            )
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
            X_mag_pad /= X_mag_pad.max()
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model.offset) // roi_size
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start : start + self.window_size]
                X_dataset.append(X_mag_window)
            X_dataset = np.asarray(X_dataset)
            self.model.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    X_batch = X_dataset[i : i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(self.device)
                    pred = self.model.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise Exception("h1_shape[3] must be greater than h2_shape[3]")
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise Exception("h1_shape[3] must be greater than h2_shape[3]")
                mask = np.concatenate(mask, axis=2)
            mask = mask[:, :, :n_frame]
            is_non_accom_stem = False
            mask = utils.adjust_aggr(mask, is_non_accom_stem, self.aggressiveness)
            v_spec = (1 - mask) * X_mag * np.exp(1.0j * X_phase)
            output_audio_path = f"{self.output_dir}/{filename}_no-noise.wav"
            source_secondary = utils.cmb_spectrogram_to_wave(
                v_spec, self.model_params, is_v51_model=self.is_vr_51_model
            ).T
            soundfile.write(
                output_audio_path,
                source_secondary,
                self.sample_rate,
                subtype=self.wav_type_set,
            )
