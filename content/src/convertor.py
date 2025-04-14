import numpy as np
from matplotlib import pyplot as plt
from bmp import BMPFormat
from wav import WAVFormat


class Convertor:
    """
    A converter class for transforming BMP images to WAV files and vice versa.
    """

    def __init__(self):
        self.bmp = BMPFormat()
        self.wav = WAVFormat()

    def generateRandomPic(self, width: int, height: int, file: str) -> str:
        """
        Generates a random RGBA image and saves it as a BMP file.

        Parameters:
            width (int): Width of the image.
            height (int): Height of the image.
            file (str): Output file path for BMP.

        Returns:
            str: The path to the saved BMP image.
        """
        pixels = np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)
        return self.bmp.saveBMP(pixel_array=pixels, path=file)

    def convert_bmpnp_to_wawnp(self, flat_pixels):
        length = len(flat_pixels)
        data = np.zeros([length])

        for pixel in range(length):
            R, G, B, A = flat_pixels[pixel]
            #print(f"{x, y}: {R, G, B, A, R<<23 or G<<15 or B<<7 or A}")
            data [pixel] =  int(((R)<<24) + ((G)<<16) + ((B)<<8) + A - (2**31-1))
            
            if data[pixel] > (2**31 -1): data[pixel] = (2**31 -1)
            elif data[pixel] < (-2**31): data[pixel] = (-2**31)
            else: pass

        return data

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
        _, _, pixel_array = self.bmp.loadBMP(path=fromFile)

        # Convert to int32 format (RGBA → single int per pixel)
        flat_pixels = pixel_array.astype(np.uint8).reshape(-1, 4)
            
        as_int32 = self.convert_bmpnp_to_wawnp(flat_pixels=flat_pixels)

        return self.wav.saveWAV(sound_data=as_int32, path=toFile,
                                 width=pixel_array.shape[1],
                                 height=pixel_array.shape[0]), pixel_array

    def wav2bmp(self, fromFile: str, toFile: str) -> str:
        """
        Converts a WAV file back into a BMP image by deserializing audio data to pixels.

        Parameters:
            fromFile (str): Path to the source WAV file.
            toFile (str): Path to the destination BMP file.

        Returns:
            tuple: A tuple containing:
                - The path to the saved BMP file.
                - soundarray
        """
        header, sound_data = self.wav.load_wav(path=fromFile)

        width = header.get("Width BMP")
        height = header.get("Height BMP")

        if width is None or height is None:
            raise ValueError("WAV file does not contain embedded image dimensions.")

        # Interpret samples as bytes → RGBA pixels
        flat_data = sound_data.astype(np.int32).tobytes()
        rgba_array = np.frombuffer(flat_data, dtype=np.uint8).reshape((height, width, 4))

        return self.bmp.saveBMP(pixel_array=rgba_array, path=toFile), sound_data
    
    def display_image(self, pixel_array, wav_data = None):
        """
        Displays the loaded BMP image using matplotlib. Optionally plots WAV data if provided.

        Args:
            pixel_array (np.ndarray): Image pixel array to be displayed.
            wav_data (np.ndarray, optional): Optional waveform data to plot below the image.

        Returns:
            None
        """
        fig, ax = plt.subplots(nrows=2, figsize=(8, 6))

        ax[0].imshow(pixel_array[::-1])
        ax[0].set_ylim(-20, pixel_array.shape[0] + 20)
        ax[0].set_xlim(-20, pixel_array.shape[1] + 20)
        ax[0].set_title("Decoded BMP Image")

        if wav_data is not None:
            ax[1].plot(np.arange(0, wav_data.shape[0], dtype=int), wav_data)

        plt.show()

    def display_wav(self, sound_data: np.ndarray, pixel_info: np.ndarray = None) -> None:
        """
        Displays the waveform of the WAV file and optional BMP reconstruction.

        Parameters:
            sound_data (np.ndarray): Decoded sound samples.
            pixel_info (np.ndarray, optional): Image data if the WAV file encodes a BMP.
        """
        fig, ax = plt.subplots(nrows=2 if pixel_info is not None else 1)

        if pixel_info is not None:
            ax[0].plot(np.arange(sound_data.shape[0]), sound_data[0])
            ax[0].set_title("Decoded WAV waveform")

            ax[1].imshow(pixel_info[::-1])
            ax[1].set_ylim(-10, pixel_info.shape[0] + 10)
            ax[1].set_xlim(-10, pixel_info.shape[1] + 10)
            ax[1].set_title("Reconstructed BMP image")
        else:
            ax.plot(np.arange(sound_data.shape[0]), sound_data)
            ax.set_title("Decoded WAV waveform")

        plt.tight_layout()
        plt.show()


