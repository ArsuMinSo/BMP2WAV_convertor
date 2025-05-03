import numpy as np
from content.src.bmp import BMPFormat
from content.src.wav import WAVFormat


class Convertor:
    """
    A converter class for transforming BMP images to WAV files and vice versa.
    """

    def __init__(self):
        self.bmp = BMPFormat()
        self.wav = WAVFormat()

    def convert_bmpnp_to_wawnp(self, flat_pixels):
        """
        Converts a flat array of pixel data (RGBA) to a flat array of audio samples (int32).
        Parameters:
            flat_pixels (np.ndarray): A flat array of pixel data in RGBA format (shape: [N, 4]).

        Returns:
            np.ndarray: A flat array of audio samples in int32 format.
        """
        flat_pixels = flat_pixels.astype(np.uint8)
        data = np.zeros((len(flat_pixels), 1), dtype=np.uint32)

        for i in range(len(flat_pixels)):
            # Convert RGBA to int32
            data[i, 0] = (flat_pixels[i, 3] & 0xFF) | \
                        ((flat_pixels[i, 2] & 0xFF) << 8) | \
                        ((flat_pixels[i, 1] & 0xFF) << 16) | \
                        ((flat_pixels[i, 0] & 0xFF) << 24)

        # Reshape to 1D array of int32
        data = data.reshape(-1)
        return data.astype(np.int32)

    def convert_wavnp_to_bmpnp(self, flat_wav):
        """
        Converts a flat array of audio samples (int32) to a flat array of pixel data (RGBA).
        """
        flat_wav = flat_wav.view(np.uint32)
        data = np.zeros((len(flat_wav), 4), dtype=np.uint8)

        for i in range(len(flat_wav)):
            # Convert int32 to RGBA
            data[i, 3] = flat_wav[i] & 0xFF
            data[i, 2] = (flat_wav[i] >> 8) & 0xFF
            data[i, 1] = (flat_wav[i] >> 16) & 0xFF
            data[i, 0] = (flat_wav[i] >> 24) & 0xFF

        return data.astype(np.int32)

    def bmp2wav(self, fromFile: str, toFile: str) -> str:
        """
        Converts a BMP image to a WAV file by serializing pixel data into audio samples.

        Parameters:
            fromFile (str): Path to the source BMP file.
            toFile (str): Path to the destination WAV file.

        Returns:
            tuple: A tuple containing:
                - The path to the saved WAV file.
                - pixelarray
        """
        _, _, pixel_array = self.bmp.load_bmp(path=fromFile)

        # Convert to int32 format (RGBA → single int per pixel)
        flat_pixels = pixel_array.astype(np.uint8).reshape(-1, 4)

        as_int32 = self.convert_bmpnp_to_wawnp(flat_pixels=flat_pixels)

        return self.wav.saveWAV(sound_data=as_int32, path=toFile,
                                width=pixel_array.shape[1],
                                height=pixel_array.shape[0]), as_int32

    def wav2bmp(self, fromFile: str, toFile: str, width: int = 0, height: int = 0) -> str:
        """
        Converts a WAV file back into a BMP image by deserializing audio data to pixels.
        """
        header, sound_data = self.wav.load_wav(path=fromFile)

        # Check if the WAV file contains a BMP header
        if width <= 0 or height <= 0:
            width = int(header.get("Width BMP"))
            height = int(header.get("Height BMP"))

        # Check if width and height are valid
        if width is None or height is None:
            raise ValueError("WAV file does not contain embedded image dimensions.")

        # Calculate the required number of samples for the image
        required_size = width * height
        current_size = len(sound_data)
        missing_data = required_size - current_size

        # If the WAV file has insufficient data, pad with zeroes
        if missing_data > 0:
            sound_data = np.pad(sound_data, (0, missing_data), mode='constant', constant_values=0)

        rgba_array = np.zeros((int(height), int(width), 4), dtype=np.uint8)
        sound_data = self.convert_wavnp_to_bmpnp(sound_data)

        for y in range(height):
            for x in range(width):
                rgba_array[y, x] = sound_data[y * width + x]

        return self.bmp.saveBMP(pixel_array=rgba_array, path=toFile), rgba_array
