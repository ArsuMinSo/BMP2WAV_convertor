import numpy as np
from numpy import sqrt

DEFAULT_FREQ = 440  # Default frequency for sine wave generation

class WAVFormat:
    def __init__(self):
        pass

    def load_wav(self, path: str) -> tuple[dict, np.ndarray]:
        """
        Loads a WAV file from the given path, extracts its header and sound data.

        Parameters:
            path (str): Path to the WAV file.

        Returns:
            tuple: A tuple containing:
                - dict: WAV header information
                - np.ndarray: Decoded sound data as NumPy array
        """
        if not path.endswith(".wav"):
            raise ValueError("Unsupported file type. Only .wav files are allowed.")

        try:
            with open(path, "rb") as f:
                raw_data = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except IOError:
            raise IOError(f"Error reading file: {path}")
        
        header = self._loadHeader(raw_data)
        sound_data = self.decode_wav(header, raw_data)
        return header, sound_data

    def _loadHeader(self, file: bytes) -> dict:
        """
        Parses the header of a WAV file and returns a dictionary of parameters.

        Parameters:
            file (bytes): Binary content of the WAV file.

        Returns:
            dict: WAV file header and custom footer (if available).
        """
        if file[:4] != b'RIFF' or file[8:12] != b'WAVE':
            raise ValueError("Invalid WAV file format.")

        header = {
            "Chunk ID": file[:4].decode("utf-8"),
            "Chunk Size": int.from_bytes(file[4:8], byteorder="little", signed=True),
            "Format": file[8:12].decode("utf-8"),

            "Subchunk 1 ID": file[12:16].decode("utf-8"),
            "Subchunk 1 size": int.from_bytes(file[16:20], byteorder="little", signed=True),
            "Audio Format": int.from_bytes(file[20:22], byteorder="little", signed=True),
            "NumChannels": int.from_bytes(file[22:24], byteorder="little", signed=True),
            "Sample Rate": int.from_bytes(file[24:28], byteorder="little", signed=True),
            "Bytes per second": int.from_bytes(file[28:32], byteorder="little", signed=True),
            "Block Align": int.from_bytes(file[32:34], byteorder="little", signed=True),
            "Bits per sample": int.from_bytes(file[34:36], byteorder="little", signed=True),
            "'Data'": file[36:40].decode("utf-8"),
            "Length": int.from_bytes(file[40:44], byteorder="little", signed=True),

            "Width BMP": int.from_bytes(file[-8:-4], byteorder="little")
            if file.find(b"Edat") != -1
            else int(sqrt(int.from_bytes(file[40:44], byteorder="little", signed=True))),
            "Height BMP": int.from_bytes(file[-4:], byteorder="little")
            if file.find(b"Edat") != -1
            else int(sqrt(int.from_bytes(file[40:44], byteorder="little", signed=True))),
        }
        return header

    def decode_wav(self, header: dict, file: bytes) -> np.ndarray:
        """
        Decodes raw audio sample data from WAV binary content using header metadata.

        Parameters:
            header (dict): Extracted WAV header information.
            file (bytes): Binary content of the WAV file.

        Returns:
            np.ndarray: Decoded audio data as NumPy array of shape (samples, channels).
        """
        sample_width = header["Bits per sample"] // 8
        num_samples = header["Length"] // sample_width
        num_channels = header["NumChannels"]

        if header["Bits per sample"] == 8:
            dtype = np.uint8
        elif header["Bits per sample"] == 16:
            dtype = np.int16
        elif header["Bits per sample"] == 24:
            dtype = np.int32
        elif header["Bits per sample"] == 32:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported bit depth: {header['Bits per sample']} bits")

        if num_channels != 1:
            raise ValueError(f"Unsupported number of channels: {num_channels} channels")
        
        sound_data = np.zeros((num_samples // num_channels), dtype=dtype)

        for i in range(0, header["Length"], sample_width * num_channels):
            for ch in range(num_channels):
                byte_index = 44 + i + ch * sample_width
                sample = int.from_bytes(
                    file[byte_index:byte_index + sample_width],
                    byteorder="little",
                    signed=True
                )
                sound_data[i // (sample_width * num_channels)] = sample

        return sound_data

    def display_wav(self, sound_data: np.ndarray, pixel_info: np.ndarray = None, show_axes=True, test_mode=False) -> None:
        """
        Displays the waveform of the WAV file and optional BMP reconstruction.

        Parameters:
            sound_data (np.ndarray): Decoded sound samples.
            pixel_info (np.ndarray, optional): Image data if the WAV file encodes a BMP.
            show_axes (bool): Whether to show axes in the plot.
            test_mode (bool): If True, saves the plot to a file instead of displaying it.
        """
        import matplotlib.pyplot as plt
        import os
        has_image = pixel_info is not None
        fig, ax = plt.subplots(nrows=2 if has_image else 1, figsize=(8, 6))

        if not has_image:
            ax = [ax]  # wrap single Axes into a list for consistency

        ax[0].plot(np.arange(sound_data.shape[0]), sound_data)
        ax[0].set_title("Decoded WAV waveform")
        if not show_axes:
            ax[0].set_xticks([])
            ax[0].set_yticks([])

        if has_image:
            ax[1].imshow(pixel_info[:, :, :3])
            ax[1].set_title("Reconstructed BMP image")
            if not show_axes:
                ax[1].set_xticks([])
                ax[1].set_yticks([])

        plt.tight_layout()

        if test_mode:
            os.makedirs("./content/media/out/test_mode/", exist_ok=True)
            plt.savefig("./content/media/out/test_mode/display_wav_output.png")
            plt.close(fig)
        else:
            plt.show()

    def saveWAV(self, sound_data: np.ndarray, path: str, width: int = None, height: int = None) -> str:
        """
        Saves raw sound data as a WAV file, optionally appending BMP dimension metadata.

        Parameters:
            sound_data (np.ndarray): Sound data to encode.
            path (str): Destination path for the WAV file.
            width (int, optional): Width of BMP image (if encoding image data).
            height (int, optional): Height of BMP image (if encoding image data).

        Returns:
            str: The path to the saved WAV file.
        """
        data = sound_data.astype(np.int32).tobytes()
        sample_rate = 44100
        num_channels = 1
        bits_per_sample = 32
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)

        header = (
            b'RIFF' +
            (36 + len(data)).to_bytes(4, 'little') +
            b'WAVE' +
            b'fmt ' +
            (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') +
            num_channels.to_bytes(2, 'little') +
            sample_rate.to_bytes(4, 'little') +
            byte_rate.to_bytes(4, 'little') +
            block_align.to_bytes(2, 'little') +
            bits_per_sample.to_bytes(2, 'little') +
            b'data' +
            len(data).to_bytes(4, 'little')
        )

        with open(path, "wb") as file:
            if width and height:
                footer = (
                    b"Edat" +
                    width.to_bytes(4, 'little') +
                    height.to_bytes(4, 'little')
                )
                file.write(header + data + footer)
            else:
                file.write(header + data)

        return path

    def print_file_info(self, info):
        """
        Prints the WAV file header information in a tabular format.
        Parameters:
            info (dict): WAV file header information.
        """
        import pandas as pd
        dataframe = pd.DataFrame.from_dict(info, orient="index", columns=["Value"])
        print(dataframe)

    def generateRandomWav(self, samples: int, file: str) -> str:
        """
        Generates a random WAV sound and saves it as a WAV file.

        Parameters:
            samples (int): Number of samples to generate.
            file (str): Output file path for BMP.

        Returns:
            str: The path to the saved BMP image.
        """
        sound_data = np.random.randint(-2**31, 2**31-1, size=(samples), dtype=np.int32)
        return self.saveWAV(sound_data, path=file)

    def generateSinWav(self, samples: int, file: str, frequency: int = DEFAULT_FREQ) -> str:
        """
        Generates a sine WAV sound and saves it as a WAV file.

        Parameters:
            samples (int): Number of samples to generate.
            frequency (int): Frequency of the sine wave.
            file (str): Output file path for WAV.

        Returns:
            str: The path to the saved WAV file.
        """
        sample_rate = 44_100  # Hz
        sound_data = np.zeros(samples, dtype=np.int32)

        for i in range(samples):
            time = i / sample_rate
            amplitude = 2**31 - 1  # max 32-bit signed int
            sound_data[i] = int(np.sin(2 * np.pi * frequency * time) * amplitude)

        return self.saveWAV(sound_data, path=file)
