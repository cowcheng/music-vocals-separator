# Music and Vocal Separator

This repository provides a command-line tool for separating music and vocals from audio files. Built as a streamlined and multithreaded adaptation of the original [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui), this tool is optimized for efficient performance with added multi-GPU support, allowing faster processing through multithreading. It supports three key models for music-vocal separation, echo reduction, and noise removal, resulting in clean vocal tracks suitable for various applications.

## Features

- **Music-Vocal Separation**: Uses the MDX23C model to separate music and vocals, saving the instrumental audio separately.
- **Echo Reduction**: Utilizes the VR DeEcho model to reduce echo effects in vocal tracks.
- **Noise Reduction**: Employs the VR Denoise model to remove background noise, producing clean vocal audio.
- **Multithreading**: Leverages multithreading to speed up processing.
- **Multi-GPU Support**: Allows the use of multiple GPUs for faster inference, optimizing the process for high-performance systems.

## Setup

### Prerequisites

- Python 3.11 or higher
- Compatible GPUs (for multi-GPU support)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/cowcheng/music-vocals-separator.git
cd music-vocals-separator
```

2. Create a virtual environment:

```bash
python3.11 -m venv .venv
```

3. Activate the virtual environment:

   1. On Windows:

   ```bash
   .venv\Scripts\activate
   ```

   2. On macOS and Linux:

   ```bash
   source .venv/bin/activate
   ```

4. Upgrade pip, wheel, and setuptools:

```bash
pip install -U pip wheel setuptools
```

5. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the tool, run the following command:

```bash
python main.py --input_dir ./input --output_dir ./output --mdx_c --de_echo --de_noise --keep_cache
```

### Command-Line Arguments

- `--input_dir`: Path to the directory containing input audio files for processing.
- `--output_dir`: Directory where the processed files (separated music and vocals) will be saved.
- `--mdx_c`: Enables the MDX23C model for music-vocal separation.
- `--de_echo`: Activates the VR DeEcho model to reduce echo in vocal tracks.
- `--de_noise`: Activates the VR Denoise model to remove background noise from vocal tracks.
- `--keep_cache`: Retains intermediate cached files generated during processing, which can be useful for debugging or re-processing.

## Workflow

1. **Music-Vocal Separation**: The MDX23C model separates the music and vocals, saving the music track as a standalone file.
2. **Echo Reduction**: The VR DeEcho model reduces echo in the vocal track, enhancing clarity.
3. **Noise Reduction**: The VR Denoise model removes background noise, producing a clean vocal file.

The final output includes both the isolated instrumental and cleaned vocal tracks, ready for archiving, mixing, or further processing.

## Notes

This tool is designed for command-line use and does not include a graphical interface.
The repository extracts code from the Ultimate Vocal Remover GUI and repurposes it for command-line functionality with added features for multithreading and multi-GPU support.

## License

Please ensure usage of this tool complies with all relevant copyright and usage laws.
