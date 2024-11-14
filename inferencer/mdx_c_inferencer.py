from typing import List

import soundfile
import torch
from tqdm import tqdm

from inferencer import utils
from inferencer.configs import MDX_C_MODEL_CONFIGS_PATH, MDX_C_MODEL_PATH
from inferencer.models.mdx_c.tfc_tdf_net import TFC_TDF_net
from inferencer.separator import Separator


class MDXCInferencer:
    def __init__(
        self,
        output_dir: str,
        device: str,
    ) -> None:
        """
        Initializes the MDX-C Inferencer.

        Args:
            output_dir (str): The directory where the output files will be saved.
            device (str): The device to be used for computation, e.g., 'cuda' or 'cpu'.

        Attributes:
            output_dir (str): The directory where the output files will be saved.
            separator (Separator): An instance of the Separator class.
            device (torch.device): The device to be used for computation.
            model_configs (dict): The configuration settings for the model.
            model (TFC_TDF_net): The neural network model for music-vocals separation.
            S (int): The number of target instruments.
            base_batch_size (int): The base batch size for processing.
            chunk_size (int): The size of each audio chunk to be processed.
            overlap (int): The overlap size for audio chunks.
            hop_size (int): The hop size for audio chunks.
            sr_pitched (int): The sample rate for pitched audio.
            is_pitch_change (bool): Flag indicating if pitch change is applied.
            sample_rate (int): The sample rate for audio processing.
            wav_type_set (str): The type of WAV file to be used.
        """

        self.output_dir = output_dir
        self.separator = Separator()
        self.device = torch.device(
            device=device if torch.cuda.is_available() else "cpu"
        )
        self.model_configs = utils.load_model_configs_file(
            config_path=MDX_C_MODEL_CONFIGS_PATH
        )
        self.model = TFC_TDF_net(
            config=self.model_configs,
            device=self.device,
        )
        self.model.load_state_dict(
            state_dict=torch.load(
                f=MDX_C_MODEL_PATH,
                map_location=torch.device(device="cpu"),
            ),
        )
        self.model.to(device=self.device).eval()
        self.S = self.model.num_target_instruments
        segment_size = 256
        self.base_batch_size = 1
        self.chunk_size = self.model_configs.audio.hop_length * (segment_size - 1)
        self.overlap = 8
        self.hop_size = self.chunk_size // self.overlap
        self.sr_pitched = 441000
        self.is_pitch_change = False
        self.sample_rate = 44100
        self.wav_type_set = "PCM_16"

    def infer(
        self,
        files_list: List[str],
    ) -> None:
        """
        Perform inference on a list of audio files to separate vocals from the music.

        This function processes each audio file in the provided list, performs vocal separation using a pre-trained model,
        and saves the separated vocals to the specified output directory. The processing includes loading the audio file,
        padding, chunking, batching, model inference, and post-processing steps such as pitch correction and saving the output.

        Args:
            files_list (List[str]): List of file paths to the audio files to be processed.

        Returns:
            None

        Note:
            - The function uses a pre-trained model for vocal separation.
            - The output files are saved in the specified output directory with '_vocals.wav' suffix.
            - The function assumes that the necessary utilities and model configurations are available as part of the class instance.
        """

        for file in tqdm(
            iterable=files_list,
            desc=f"MDX-C inferencing on device {self.device}",
        ):
            filename = file.split("/")[-1].split(".")[0]
            audio_npy = utils.load_audio_file(audio_path=file)
            org_audio_npy = audio_npy
            audio_t = torch.tensor(
                data=audio_npy,
                dtype=torch.float32,
            )
            audio_t_shape = audio_t.shape[1]
            pad_size = self.hop_size - (audio_t_shape - self.chunk_size) % self.hop_size
            audio_t = torch.cat(
                [
                    torch.zeros(2, self.chunk_size - self.hop_size),
                    audio_t,
                    torch.zeros(2, pad_size + self.chunk_size - self.hop_size),
                ],
                1,
            )
            chunks = audio_t.unfold(1, self.chunk_size, self.hop_size).transpose(0, 1)
            batches = [
                chunks[i : i + self.base_batch_size]
                for i in range(0, len(chunks), self.base_batch_size)
            ]
            X = (
                torch.zeros(self.S, *audio_t.shape)
                if self.S > 1
                else torch.zeros_like(audio_t)
            )
            X = X.to(self.device)
            with torch.no_grad():
                cnt = 0
                for batch in batches:
                    x = self.model(batch.to(self.device))
                    for w in x:
                        X[
                            ...,
                            cnt * self.hop_size : cnt * self.hop_size + self.chunk_size,
                        ] += w
                        cnt += 1
            estimated_sources = (
                X[
                    ...,
                    self.chunk_size
                    - self.hop_size : -(pad_size + self.chunk_size - self.hop_size),
                ]
                / self.overlap
            )
            del X
            pitch_fix = lambda s: self.separator.pitch_fix(
                s, self.sr_pitched, org_audio_npy
            )
            sources = {
                k: pitch_fix(v) if self.is_pitch_change else v
                for k, v in zip(
                    self.model_configs.training.instruments,
                    estimated_sources.cpu().detach().numpy(),
                )
            }
            del estimated_sources
            source_primary = sources["Vocals"]
            output_audio_path = f"{self.output_dir}/{filename}_vocals.wav"
            source_primary = source_primary.T
            soundfile.write(
                file=output_audio_path,
                data=source_primary,
                samplerate=self.sample_rate,
                subtype=self.wav_type_set,
            )
