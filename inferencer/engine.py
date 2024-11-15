import glob
import os
import uuid
import warnings
from threading import Thread
from typing import List, Type

import torch

from inferencer.configs import ROOT_DIR
from inferencer.mdx_c_inferencer import MDXCInferencer
from inferencer.utils import logger
from inferencer.vr_de_echo_inferencer import VRDeEchoInferencer
from inferencer.vr_de_noise_inferencer import VRDeNoiseInferencer

warnings.filterwarnings(action="ignore")


class MusicVocalsSeparatorEngine:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        _use_mdx_c: bool,
        _use_vr_de_echo: bool,
        _use_vr_de_noise: bool,
        _keep_cache: bool,
    ) -> None:
        """
        Initializes the Music Vocals Separator Engine.

        Args:
            input_dir (str): Path to the input directory containing audio files.
            output_dir (str): Path to the output directory where results will be saved.
            _use_mdx_c (bool): Specifies whether to use the MDX-C model.
            _use_vr_de_echo (bool): Specifies whether to use the VR De-Echo model.
            _use_vr_de_noise (bool): Specifies whether to use the VR Denoise model.
            _keep_cache (bool): Specifies whether to keep the cache.

        Raises:
            AssertionError: If no GPUs are available.
        """

        self.input_dir = input_dir
        self.output_dir = output_dir
        self._use_mdx_c = _use_mdx_c
        self._use_vr_de_echo = _use_vr_de_echo
        self._use_vr_de_noise = _use_vr_de_noise
        self._keep_cache = _keep_cache

        logger.info(msg="Initializing Music Vocals Separator Engine...")
        logger.info(msg=f"Use MDX-C model: {self._use_mdx_c}")
        logger.info(msg=f"Use DE-Echo model: {self._use_vr_de_echo}")
        logger.info(msg=f"Use Denoise model: {self._use_vr_de_noise}")
        logger.info(msg=f"Keep cache: {self._keep_cache}")
        logger.info(msg=f"Input directory: {self.input_dir}")
        logger.info(msg=f"Output directory: {self.output_dir}")

        self.batch_id = uuid.uuid4().hex
        logger.info(msg=f"Batch ID: {self.batch_id}")

        self.num_gpus = torch.cuda.device_count()
        assert self.num_gpus > 0, "No GPUs available"
        logger.info(msg=f"Number of GPUs available: {self.num_gpus}")

        self.inferencers = []
        self.tmp_dirs = {}
        self.last_tmp_dir = None

        if self._use_mdx_c:
            self._initialize_inferencers(
                inferencer_cls=MDXCInferencer,
                tmp_dir=f"{ROOT_DIR}/tmp/{self.batch_id}/mdx_c",
                key="mdx_c",
            )
        if self._use_vr_de_echo:
            self._initialize_inferencers(
                inferencer_cls=VRDeEchoInferencer,
                tmp_dir=f"{ROOT_DIR}/tmp/{self.batch_id}/vr_de_echo",
                key="vr_de_echo",
            )
        if self._use_vr_de_noise:
            self._initialize_inferencers(
                inferencer_cls=VRDeNoiseInferencer,
                tmp_dir=f"{ROOT_DIR}/tmp/{self.batch_id}/vr_de_noise",
                key="vr_de_noise",
            )

    def _initialize_inferencers(
        self,
        inferencer_cls: Type,
        tmp_dir: str,
        key: str,
    ) -> None:
        """
        Initialize inferencer instances on available GPUs.

        This method sets up temporary directories and initializes instances of the specified inferencer class on each GPU.

        Args:
            inferencer_cls (Type): The inferencer class to instantiate.
            tmp_dir (str): The temporary directory path for output.
            key (str): A key to identify the temporary directory.

        Returns:
            None
        """

        self.tmp_dirs[key] = tmp_dir
        self.last_tmp_dir = key
        os.makedirs(
            name=tmp_dir,
            exist_ok=True,
        )
        for gpu_id in range(self.num_gpus):
            logger.info(
                msg=f"Initializing {inferencer_cls.__name__} on GPU {gpu_id}..."
            )
            inferencer = inferencer_cls(
                output_dir=tmp_dir,
                device=f"cuda:{gpu_id}",
            )
            self.inferencers.append(inferencer)

    def run(self) -> None:
        """
        Executes the music vocal separation process.

        This method performs the following steps:
            1. Collects all audio files from the input directory.
            2. Processes the audio files using multiple inferencers:
                - MDX C
                - VR De Echo
                - VR De Noise
            3. Renames the last temporary directory to the output directory.
            4. Cleans up temporary files if caching is not enabled.

        Logs information about the process flow and completion status.

        Returns:
            None
        """

        audio_files = glob.glob(pathname=f"{self.input_dir}/*.*")
        logger.info(msg=f"Number of audio files found: {len(audio_files)}")
        self._process_inferencers(
            files_list=audio_files,
            _use=self._use_mdx_c,
        )
        self._process_inferencers(
            files_list=glob.glob(pathname=f"{self.tmp_dirs['mdx_c']}/*.*"),
            _use=self._use_vr_de_echo,
        )
        self._process_inferencers(
            files_list=glob.glob(pathname=f"{self.tmp_dirs['vr_de_echo']}/*.*"),
            _use=self._use_vr_de_noise,
        )

        logger.info(
            msg="Renaming the last temporary directory to the output directory..."
        )
        if self.last_tmp_dir:
            os.rename(
                src=self.tmp_dirs[self.last_tmp_dir],
                dst=self.output_dir,
            )
        if not self._keep_cache:
            os.remove(path=f"{ROOT_DIR}/tmp/{self.batch_id}")
        logger.info(msg="Music Vocals Separator Engine completed")

    def _process_inferencers(
        self,
        files_list: List,
        _use: bool,
    ) -> None:
        """
        Processes audio files using inferencers across multiple GPUs.

        This method distributes the provided audio files evenly across the available GPUs.
        For each GPU, it assigns the corresponding inferencer to process its assigned files
        in a separate thread. After processing, it cleans up the inferencers and clears the CUDA cache.

        Args:
            files_list (List): A list of audio file paths to be processed.
            _use (bool): Flag indicating whether to use the inferencers.

        Returns:
            None
        """

        if _use:
            audio_files_per_gpu = [[] for _ in range(self.num_gpus)]
            for idx, file in enumerate(files_list):
                audio_files_per_gpu[idx % self.num_gpus].append(file)
            threads = []
            for gpu_id, assigned_files in enumerate(audio_files_per_gpu):
                print(
                    f"Assigning {len(assigned_files)} files to {self.inferencers[gpu_id]} on GPU {gpu_id}"
                )
                thread = Thread(
                    target=self.inferencers[gpu_id].infer,
                    args=(assigned_files,),
                )
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            del self.inferencers[: self.num_gpus]
            torch.cuda.empty_cache()
