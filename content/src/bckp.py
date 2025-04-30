import numpy as np
from matplotlib import pyplot as plt
from numpy import sqrt
import argparse

class BMPFormat:
    def __init__(self):
        """Initializes the BMPFormat class for handling BMP file loading and saving."""
        pass

    def load_bmp(self, path: str) -> tuple:
        """
        Loads a BMP image from the specified file path and decodes it into a NumPy array.

        Args:
            path (str): Path to the BMP file.

        Returns:
            tuple: A tuple containing:
                - dict: Header information.
                - np.ndarray: Color palette (if applicable), otherwise None.
                - np.ndarray: Pixel data as a 3D NumPy array with shape (H, W, 4) in RGBA format.

        Raises:
            Exception: If file extension is not '.bmp' or the BMP format is unsupported.
        """
        if not path.endswith(".bmp"):
            raise ValueError("Unsupported file type. Only .bmp files are allowed.")

        with open(path, "rb") as file:
            header_offset = 54
            bmp_header = self.loadHeader(file.read(header_offset))

            palette_offset = bmp_header["Palette offset"] - header_offset
            palette = self.loadPalette(file.read(palette_offset), bmp_header)
            pixel_plane = self.convert_bmp_to_numpy(file.read(), palette, header=bmp_header)
            return bmp_header, palette, pixel_plane

    def loadHeader(self, file: bytearray) -> dict:
        """
        Parses the BMP file header (first 54 bytes) and extracts metadata.

        Args:
            file (bytearray): The first 54 bytes of a BMP file.

        Returns:
            dict: Extracted BMP header fields with their corresponding values.

        Raises:
            Exception: If the BMP header size is not 40 bytes (BITMAPINFOHEADER).
        """
        header_size = int.from_bytes(file[14:18], byteorder="little", signed=True)

        if header_size != 40:
            raise Exception("Unsupported BMP format, only BMP with standard header is supported.")

        header = {
            "BMP identifier": file[:2].decode("utf-8"),
            "File size": int.from_bytes(file[2:6], byteorder="little", signed=True),
            "Reserved bytes": int.from_bytes(file[6:10], byteorder="little", signed=True),
            "Palette offset": int.from_bytes(file[10:14], byteorder="little", signed=True),
            "Header size": int.from_bytes(file[14:18], byteorder="little", signed=True),
            "Width": int.from_bytes(file[18:22], byteorder="little", signed=True),
            "Height": int.from_bytes(file[22:26], byteorder="little", signed=True),
            "Planes": int.from_bytes(file[26:28], byteorder="little", signed=True),
            "Bits per pixel": int.from_bytes(file[28:30], byteorder="little", signed=True),
            "Compression": int.from_bytes(file[30:34], byteorder="little", signed=True),
            "Image size": int.from_bytes(file[34:38], byteorder="little", signed=True),
            "Horizontal resolution": round(int.from_bytes(file[38:42], byteorder="little", signed=True)/100 * 2.54, 0),
            "Vertical resolution": round(int.from_bytes(file[42:46], byteorder="little", signed=True)/100 * 2.54, 0),
            "Number of colors in the palette": int.from_bytes(file[46:50], byteorder="little", signed=True),
            "Important colors": int.from_bytes(file[50:54], byteorder="little", signed=True),
        }

        return header

    def loadPalette(self, file, header) -> np.array:
        """
        Extracts the color palette from BMP data (used in 1, 4, and 8-bit BMPs).

        Args:
            file (bytes): Raw palette data.
            header (dict): BMP header containing information like bits per pixel.

        Returns:
            np.ndarray: NumPy array of shape (N, 4) containing RGBA values of the palette,
                        or None if the image doesn't use a palette.
        """
        if header["Bits per pixel"] in [1, 4, 8]:
            palette_hex = file

            num_colors = header["Number of colors in the palette"] or 2 ** header["Bits per pixel"]
            palette = np.zeros([num_colors, 4], dtype=np.uint8)

            for i in range(0, len(palette_hex), 4):
                B = palette_hex[i]
                G = palette_hex[i + 1]
                R = palette_hex[i + 2]
                U = palette_hex[i + 3]
                palette[i // 4] = [R, G, B, 255 - U]
        else:
            palette = None

        return palette

    def convert_bmp_to_numpy(self, file, palette, header):
        """
        Converts the pixel data of a BMP image into a NumPy array.

        Args:
            file (bytes): Raw BMP pixel data.
            palette (np.ndarray or None): Color palette (if applicable).
            header (dict): Parsed BMP header.

        Returns:
            np.ndarray: Image pixel array in RGBA format.
        """
        pixel_data = file
        width, height = header["Width"], header["Height"]
        bpp = header["Bits per pixel"]

        scanline_size = self.get_scanline_size(width, bpp)

        if bpp in [1, 4, 8]:
            pixel_info = self.decode_indexed_image(pixel_data, palette, width, height, bpp, scanline_size)
        elif bpp in [24, 32]:
            pixel_info = self.decode_direct_image(pixel_data, width, height, bpp, scanline_size)

        return pixel_info

    def get_scanline_size(self, width, bpp):
        """
        Computes the size of one scanline in bytes (padded to 4-byte boundaries).

        Args:
            width (int): Image width in pixels.
            bpp (int): Bits per pixel.

        Returns:
            int: Number of bytes per scanline.
        """
        bits_per_row = width * bpp
        scanline = ((bits_per_row + 31) // 32) * 4

        return scanline

    def decode_indexed_image(self, pixel_data, palette, width, height, bpp, scanline_size):
        """
        Decodes an indexed BMP image (1, 4, or 8 bits per pixel) into a full-color image.

        Args:
            pixel_data (bytes): Encoded pixel data.
            palette (np.ndarray): Palette used to map indices to colors.
            width (int): Image width.
            height (int): Image height.
            bpp (int): Bits per pixel.
            scanline_size (int): Number of bytes per scanline.

        Returns:
            np.ndarray: Decoded image as a 3D NumPy array (H, W, 4).
        """
        pixel_plane = np.zeros([height, width, 4], dtype=np.uint8)

        byte_index = 0
        for y in range(height):
            print(y)
            row_data = pixel_data[byte_index:byte_index + scanline_size]

            if bpp == 8:
                row_data = row_data[:width]  # strip padding

            row_pixels = self.decode_indexed_row(row_data, width, bpp, palette)
            pixel_plane[height - 1 - y] = row_pixels
            byte_index += scanline_size

        return pixel_plane

    def decode_indexed_row(self, row_data, width, bpp, palette_info):
        """
        Decodes one scanline of an indexed BMP image into RGBA pixels.

        Args:
            row_data (bytes): One row of encoded pixel data.
            width (int): Number of pixels in the row.
            bpp (int): Bits per pixel (1, 4, or 8).
            palette_info (np.ndarray): Palette used for decoding.

        Returns:
            np.ndarray: Row of pixels as RGBA values.
        """
        row = np.zeros([width, 4], dtype=np.uint8)
        byte_index = 0
        bit_offset = 0

        for x in range(width):
            if bpp == 8:
                print(f"width: {width}, len(row_data): {len(row_data)}")

                index = row_data[x]
            else:
                if bit_offset == 0:
                    current_byte = row_data[byte_index]
                    byte_index += 1

                if bpp == 4:
                    index = (current_byte >> (4 if bit_offset == 0 else 0)) & 0x0F
                    bit_offset = (bit_offset + 4) % 8
                elif bpp == 1:
                    index = (current_byte >> (7 - bit_offset)) & 0x01
                    bit_offset = (bit_offset + 1) % 8

            row[x] = palette_info[index]

        return row

    def decode_direct_image(self, pixel_data, width, height, bpp, scanline_size):
        """
        Decodes a BMP image that stores direct RGB(A) values (24 or 32 bits per pixel).

        Args:
            pixel_data (bytes): Encoded image data.
            width (int): Image width.
            height (int): Image height.
            bpp (int): Bits per pixel (24 or 32).
            scanline_size (int): Number of bytes per scanline.

        Returns:
            np.ndarray: Image in RGBA format.
        """
        pixel_plane = np.zeros([height, width, 4], dtype=np.uint8)

        byte_index = 0
        for y in range(height):
            row_pixels = self.decode_direct_row(pixel_data[byte_index:byte_index + scanline_size], width, bpp)
            pixel_plane[height - 1 - y] = row_pixels
            byte_index += scanline_size

        return pixel_plane

    def decode_direct_row(self, row_data, width, bpp):
        """
        Decodes one scanline of a direct color BMP image.

        Args:
            row_data (bytes): Raw row data.
            width (int): Number of pixels in the row.
            bpp (int): Bits per pixel (24 or 32).

        Returns:
            np.ndarray: Row of pixels in RGBA format.
        """
        row = np.zeros([width, 4], dtype=np.uint8)
        bytes_per_pixel = bpp // 8

        for x in range(width):
            pixel_offset = x * bytes_per_pixel
            B, G, R = row_data[pixel_offset:pixel_offset + 3]
            A = row_data[pixel_offset + 3] if bpp == 32 else 255
            row[x] = [R, G, B, A]

        return row

    def saveBMP(self, pixel_array, path) -> str:
        """
        Saves a 24-bit BMP image to disk from a pixel array (ignores alpha channel).

        Args:
            pixel_array (np.ndarray): A NumPy array with shape (H, W, 3 or 4).
            path (str): Output file path.

        Returns:
            str: Path to the saved file.
        """
        with open(path, "wb") as file:
            bmp_data = b""

            height, width, depth = pixel_array.shape  # POZOR: pořadí je (H, W, C)
            row_padding = (4 - (width * 3) % 4) % 4  # BMP s 24 bpp (bez alpha), 3 bajty na pixel

            for y in range(height - 1, -1, -1):  # BMP ukládá řádky odspodu
                row_data = b"".join(
                    pixel_array[y, x, 2].tobytes() +  # Blue
                    pixel_array[y, x, 1].tobytes() +  # Green
                    pixel_array[y, x, 0].tobytes()    # Red
                    for x in range(width)
                )
                bmp_data += row_data + (b"\x00" * row_padding)

            file_size = 54 + len(bmp_data)
            bmp_header = (
                b'BM' +
                file_size.to_bytes(4, 'little') +
                b'\x00\x00' + b'\x00\x00' +
                (54).to_bytes(4, 'little') +  # Offset to pixel data
                (40).to_bytes(4, 'little') +  # DIB header size
                width.to_bytes(4, 'little') +
                height.to_bytes(4, 'little') +
                (1).to_bytes(2, 'little') +   # Color planes
                (24).to_bytes(2, 'little') +  # Bits per pixel
                (0).to_bytes(4, 'little') +   # No compression
                len(bmp_data).to_bytes(4, 'little') +
                (2835).to_bytes(4, 'little') +  # Horizontal resolution
                (2835).to_bytes(4, 'little') +  # Vertical resolution
                (0).to_bytes(4, 'little') +     # Colors in palette
                (0).to_bytes(4, 'little')       # Important colors
            )

            file.write(bmp_header + bmp_data)
            file.close()
            return path
        return ""

    def display_image(self, bmp_data, wav_data=None):
        """
        Displays the loaded BMP image using matplotlib. Optionally plots WAV data if provided.

        Args:
            bmp_data (np.ndarray): Image pixel array to be displayed.
            wav_data (np.ndarray, optional): Optional waveform data to plot below the image.

        Returns:
            None
        """
        fig, ax = plt.subplots(nrows=2 if wav_data is not None else 1)

        if wav_data is not None:
            ax[0].imshow(bmp_data[::-1])
            ax[0].set_ylim(-20, bmp_data.shape[0] + 20)
            ax[0].set_xlim(-20, bmp_data.shape[1] + 20)
            ax[0].set_title("Decoded BMP Image")

            ax[1].plot(np.arange(0, wav_data.shape[0], dtype=int), wav_data)
            ax[1].set_title("Reconstructed WAV wave")

        else:
            ax.imshow(bmp_data[::-1])
            ax.set_ylim(-20, bmp_data.shape[0] + 20)
            ax.set_xlim(-20, bmp_data.shape[1] + 20)
            ax.set_title("Decoded BMP Image")

        plt.tight_layout()
        plt.show()

    def print_file_info(self, info):
        """
        Prints the BMP file information in a tabular format using pandas.

        Parameters:
            info (dict): Dictionary containing BMP file information.
        """
        import pandas as pd
        dataframe = pd.DataFrame.from_dict(info, orient="index", columns=["Value"])
        print(dataframe)

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
        return self.saveBMP(pixel_array=pixels, path=file)


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

        with open(path, "rb") as f:
            raw_data = f.read()

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
            "RIFF Header": file[:4].decode("utf-8"),
            "File size": int.from_bytes(file[4:8], byteorder="little", signed=True),
            "WAVE": file[8:12].decode("utf-8"),

            "Signature 'fmt '": file[12:16].decode("utf-8"),
            "Header size": int.from_bytes(file[16:20], byteorder="little", signed=True),
            "Format Tag": int.from_bytes(file[20:22], byteorder="little", signed=True),
            "Channels": int.from_bytes(file[22:24], byteorder="little", signed=True),
            "Sample Rate": int.from_bytes(file[24:28], byteorder="little", signed=True),
            "Bytes per second": int.from_bytes(file[28:32], byteorder="little", signed=True),
            "Block size": int.from_bytes(file[32:34], byteorder="little", signed=True),
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
        print(header["Width BMP"], header["Height BMP"], header["Length"])
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
        num_channels = header["Channels"]

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

        sound_data = np.zeros((num_samples // num_channels, num_channels), dtype=dtype)

        for i in range(0, header["Length"], sample_width * num_channels):
            for ch in range(num_channels):
                byte_index = 44 + i + ch * sample_width
                sample = int.from_bytes(
                    file[byte_index:byte_index + sample_width],
                    byteorder="little",
                    signed=True
                )
                sound_data[i // (sample_width * num_channels), ch] = sample

        return sound_data

    def display_wav(self, sound_data: np.ndarray, pixel_info: np.ndarray = None) -> None:
        """
        Displays the waveform of the WAV file and optional BMP reconstruction.

        Parameters:
            sound_data (np.ndarray): Decoded sound samples.
            pixel_info (np.ndarray, optional): Image data if the WAV file encodes a BMP.
        """
        fig, ax = plt.subplots(nrows=2 if pixel_info is not None else 1)

        if pixel_info is not None:
            ax[0].plot(np.arange(sound_data.shape[0]), sound_data)
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

    def generateSinWav(self, samples: int, frequency: int, file: str) -> str:
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
            flat_pixels (np.ndarray): A flat array of pixel data in RGBA format.

        Returns:
            np.ndarray: A flat array of audio samples in int32 format.
        """
        length = len(flat_pixels)
        data = np.zeros([length])

        # Convert RGBA to int32 format
        for pixel in range(length):
            R, G, B, A = flat_pixels[pixel]
            data[pixel] = int(((R) << 24) + ((G) << 16) + ((B) << 8) + A - (2**31 - 1))

            if data[pixel] > (2**31 - 1):
                data[pixel] = (2**31 - 1)
            elif data[pixel] < (-2**31):
                data[pixel] = (-2**31)
            else:
                pass

        return data

    def convert_wavnp_to_bmpnp(self, flat_wav):
        """
        Converts a flat array of audio samples (int32) to a flat array of pixel data (RGBA).
        Parameters:
            flat_wav (np.ndarray): A flat array of audio samples in int32 format.

        Returns:
            np.ndarray: A flat array of pixel data in RGBA format.
        """
        length = len(flat_wav)
        data = np.zeros([length, 4])

        # Convert int32 to RGBA format
        # Note: The conversion assumes that the audio samples are in the range of int32.
        for sample in range(length):
            data[sample][0] = ((flat_wav[sample] >> 24) & 0xFF) + (128 if flat_wav[sample] >= 0 else -128)
            data[sample][1] = (flat_wav[sample] >> 16) & 0xFF
            data[sample][2] = (flat_wav[sample] >> 8) & 0xFF
            data[sample][3] = (flat_wav[sample] >> 0) & 0xFF
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

    def wav2bmp(self, fromFile: str, toFile: str, width: int = 0, height: int = 0) -> str:
        """
        Converts a WAV file back into a BMP image by deserializing audio data to pixels.

        Parameters:
            fromFile (str): Path to the source WAV file.
            toFile (str): Path to the destination BMP file.
            width (int): Default 0, width of converted picture
            height (int): Default 0, heiht of converted picture

        Returns:
            tuple: A tuple containing:
                - The path to the saved BMP file.
                - soundarray
        """
        header, sound_data = self.wav.load_wav(path=fromFile)

        # Check if the WAV file contains a BMP header
        if width <= 0 or height <= 0:
            width = int(header.get("Width BMP"))
            height = int(header.get("Height BMP"))

        # Check if width and height are valid
        if width is None or height is None:
            raise ValueError("WAV file does not contain embedded image dimensions.")

        # Interpret samples as bytes → RGBA pixels
        rgba_array = np.zeros((int(height), int(width), 4), dtype=np.uint8)
        sound_data = self.convert_wavnp_to_bmpnp(sound_data)
        for y in range(height):
            for x in range(width):
                try:
                    rgba_array[y, x] = sound_data[y * width + x]
                except ValueError:
                    pass

        return self.bmp.saveBMP(pixel_array=rgba_array, path=toFile), rgba_array


def main():
    parser = argparse.ArgumentParser(
        prog='BMP ↔ WAV Convertor',
        description='Converts BMP images to WAV files and vice versa, or generates random images.',
        epilog='Created by ArsuMinSo'
    )

    parser.add_argument('-gb', '--generate-bmp',
                        nargs=3,
                        metavar=('WIDTH', 'HEIGHT', 'OUTFILE'),
                        help='Generate random BMP image.')

    parser.add_argument('-gw', '--generate-wav',
                        nargs=2,
                        metavar=('SAMPLES', 'OUTFILE'),
                        help='Generate random WAV wave. If ISSINE is true, generates sine wave')

    parser.add_argument('--freq',
                        type=int,
                        nargs=1,
                        metavar=('FREQENCY'),
                        default=None,
                        help='Frequency of the sine wave - used with generate-wav (default: 440Hz)')

    parser.add_argument('-c', '--convert',
                        nargs=2,
                        metavar=('INFILE', 'OUTFILE'),
                        help='Convert one format to another.')

    parser.add_argument('-d', '--dimensions',
                        type=int,
                        nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        default=(0, 0),
                        help='Dimensions of picture for wav to bmp convertion')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose mode.')

    parser.add_argument('-i', '--info',
                        nargs=1,
                        metavar=('FILE'),
                        help='Shows metadata about file')

    parser.add_argument('--show',
                        action='store_true',
                        help='Show plots in matplotlib.')

    args = parser.parse_args()
    convertor = Convertor()

    if args.generate_wav:
        samples, outfile = args.generate_wav
        samples = int(samples)
        freq = args.freq[0]
        if not freq:
            result = convertor.wav.generateRandomWav(samples, outfile)
        else:
            result = convertor.generateSinWav(samples, freq, outfile)

        if args.verbose:
            print(f"Generated wav saved to {result}")
        if args.show:
            wav_data = convertor.wav.load_wav(result)
            convertor.wav.display_wav(wav_data[1])

    elif args.generate_bmp:
        width, height, outfile = args.generate_bmp
        width = int(width)
        height = int(height)
        result = convertor.bmp.generateRandomPic(width, height, outfile)
        if args.verbose:
            print(f"Generated image saved to {result}")
        if args.show:
            pixel_data = convertor.bmp.load_bmp(result)
            convertor.bmp.display_image(pixel_data[2])

    elif args.convert:
        input_file, output_file = args.convert

        if input_file.endswith(".bmp") and output_file.endswith(".wav"):
            if args.verbose:
                print(f"Reading BMP from {input_file}")

            bmp_array = convertor.bmp.load_bmp(input_file)

            if args.verbose:
                print(f"Saving WAV to {output_file}")

            wav_data = convertor.bmp2wav(input_file, output_file)

            print(f"Converted BMP to WAV: {output_file}")
            if args.show:
                convertor.bmp.display_image(bmp_array[2], wav_data[1])

        elif input_file.endswith(".wav") and output_file.endswith(".bmp"):
            width, height = args.dimensions
            if args.verbose:
                print(f"Reading WAV from {input_file}")

            wav_data = convertor.wav.load_wav(input_file)

            if args.verbose:
                print(f"Saving BMP to {output_file}")

            bmp_array = convertor.wav2bmp(input_file, output_file, width, height)

            print(f"Converted WAV to BMP: {output_file}")
            if args.show:
                convertor.wav.display_wav(wav_data[1], bmp_array[1])

        else:
            raise ValueError("Wrong files")

    elif args.info:
        input_file = args.info[0]

        if input_file.endswith(".bmp"):
            if args.verbose:
                print(f"Reading BMP from {input_file}")

            bmp_array = convertor.bmp.load_bmp(input_file)
            convertor.bmp.print_file_info(bmp_array[0])

            if args.show:
                convertor.bmp.display_image(bmp_array[2])

        elif input_file.endswith(".wav"):
            if args.verbose:
                print(f"Reading WAV from {input_file}")

            wav_data = convertor.wav.load_wav(input_file)
            convertor.wav.print_file_info(wav_data[0])

            if args.show:
                convertor.wav.display_wav(wav_data[1])
        else:
            raise ValueError("Wrong files")

    if args.verbose:
        print("Verbose mode enabled")
        print("All arguments:", args)

