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
    
    def generateRandomWav(self, samples: int, file: str) -> str:
        """
        Generates a random RGBA image and saves it as a BMP file.

        Parameters:
            samples (int): Number of samples to generate.
            file (str): Output file path for BMP.

        Returns:
            str: The path to the saved BMP image.
        """
        sound_data = np.random.randint(-2**31, 2**31-1, size=(samples), dtype=np.int32)
        return self.wav.saveWAV(sound_data, path=file)

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
    
    def convert_wavnp_to_bmpnp(self, flat_wav):
        length = len(flat_wav)
        data = np.zeros([length, 4])

        for sample in range(length):
            data [sample][0] = ((flat_wav[sample] >> 24) & 0xFF) + (128 if flat_wav[sample]>=0 else -128)
            data [sample][1] =  (flat_wav[sample] >> 16) & 0xFF
            data [sample][2] =  (flat_wav[sample] >>  8) & 0xFF
            data [sample][3] =  254#(flat_wav[sample] >>  0) & 0xFF
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
        _, _, pixel_array = self.bmp.load_bmp(path=fromFile)

        # Convert to int32 format (RGBA → single int per pixel)
        flat_pixels = pixel_array.astype(np.uint8).reshape(-1, 4)
            
        as_int32 = self.convert_bmpnp_to_wawnp(flat_pixels=flat_pixels)

        return self.wav.saveWAV(sound_data=as_int32, path=toFile,
                                 width=pixel_array.shape[1],
                                 height=pixel_array.shape[0]), as_int32

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
        rgba_array = np.array(self.convert_wavnp_to_bmpnp(sound_data).reshape((height, width, 4)), dtype=np.uint8)

        return self.bmp.saveBMP(pixel_array=rgba_array, path=toFile), rgba_array