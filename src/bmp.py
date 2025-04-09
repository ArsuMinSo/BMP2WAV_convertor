import numpy as np
from numpy import sqrt, ceil
import math
from matplotlib import pyplot as plt

class BMPFormat:
    def __init__(self):
        pass


    def loadBMP(self, path:str) -> tuple:
        if path.endswith(".bmp"):
            with open(path, "rb") as file:
                header_offset = 54
                bmp_header = self._loadHeader(file.read(header_offset))

                palette_offset = bmp_header["Palette offset"] - header_offset
                palette = self._loadPalette(file.read(palette_offset), bmp_header)

                
                pixel_plane = self._convert_bmp_to_numpy(file.read(), palette, header=bmp_header)
                
        return (bmp_header, palette, pixel_plane)


    def _loadHeader(self, file:bytearray) -> dict:
        """
        Reads and extracts header information from the BMP file.
        Only supports standard BMP headers of size 40 bytes.
        """
        header_size = int.from_bytes(file[14:18], byteorder="little", signed=True)
        
        if header_size != 40:
            raise Exception("Unsupported BMP format, only 1, 2, 4, 8 or 24-bit BMP with standard header is supported.")

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
            "Horizontal resolution": round(int.from_bytes(file[38:42], byteorder="little", signed=True)/100 *2.54, 0),
            "Vertical resolution": round(int.from_bytes(file[42:46], byteorder="little", signed=True)/100 *2.54, 0),
            "Number of colors in the palette": int.from_bytes(file[46:50], byteorder="little", signed=True),
            "Important colors": int.from_bytes(file[50:54], byteorder="little", signed=True),
        }

        return header


    def _loadPalette(self, file, header) -> np.array:
        """
        Extracts the color palette from BMP file if applicable (for 1, 4, and 8-bit BMPs).
        """
        if header["Bits per pixel"] in [1, 4, 8]:
            palette_hex = file

            num_colors = header["Number of colors in the palette"] or 2 ** header["Bits per pixel"]
            palette = np.zeros([num_colors, 4], dtype=np.uint8)

            for i in range(0, len(palette_hex), 4):
                B, G, R, U  = palette_hex[i], palette_hex[i+1], palette_hex[i+2], palette_hex[i+3]
                palette[i//4] = [R, G, B, 255 - U]
        else:
            palette = None

        return palette


    def _convert_bmp_to_numpy(self, file, palette, header):
        """Converts BMP pixel data into a NumPy array."""
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
        """Calculates the BMP scanline size (padded to 4-byte boundaries)."""
        bits_per_row = width * bpp
        scanline = ((bits_per_row + 31) // 32) * 4
        
        return scanline


    def decode_indexed_image(self, pixel_data, palette, width, height, bpp, scanline_size):
        """Decodes BMP images with a color palette (1, 4, 8-bit) using scanlines."""
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
        """Decodes a single scanline for indexed BMP images."""
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
        """Decodes BMP images without a palette (24, 32-bit) using scanlines."""
        pixel_plane = np.zeros([height, width, 4], dtype=np.uint8)

        byte_index = 0
        for y in range(height):
            row_pixels = self.decode_direct_row(pixel_data[byte_index:byte_index + scanline_size], width, bpp)
            pixel_plane[height - 1 - y] = row_pixels
            byte_index += scanline_size

        return pixel_plane

    
    def decode_direct_row(self, row_data, width, bpp):
        """Decodes a single scanline for direct color BMP images."""
        row = np.zeros([width, 4], dtype=np.uint8)
        bytes_per_pixel = bpp // 8

        for x in range(width):
            pixel_offset = x * bytes_per_pixel
            B, G, R = row_data[pixel_offset:pixel_offset + 3]
            A = row_data[pixel_offset + 3] if bpp == 32 else 255
            row[x] = [R, G, B, A]

        return row
    
    def saveBMP(self, pixel_array, path) -> str:
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


    def display_image(self, pixel_array, wav_data = None):
        """Displays the decoded BMP image using Matplotlib."""
        
        fig, ax = plt.subplots(nrows=2, figsize=(8, 6))

        ax[0].imshow(pixel_array[::-1])
        ax[0].set_ylim(-20, pixel_array.shape[0] + 20)
        ax[0].set_xlim(-20, pixel_array.shape[1] + 20)
        ax[0].set_title("Decoded BMP Image")

        if wav_data != None:
            ax[1].plot(np.arange(0, wav_data.shape[0], dtype=int), wav_data)

        plt.show()

