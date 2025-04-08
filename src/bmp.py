import numpy as np
from numpy import sqrt, ceil
import math
from matplotlib import pyplot as plt


class BMPImage:
    def __init__(self):
        pass

    def loadBMP(self, path):
        if path.endswith(".bmp"):
            with open(path, "rb") as file:
                header = file.read(54)
                bmp_header = self._loadHeader(header)
                # palette = self._loadPalette(file, bmp_header)
        return bmp_header

    def _loadHeader(self, file) -> dict:
        """
        Reads and extracts header information from the BMP file.
        Only supports standard BMP headers of size 40 bytes.
        """
        header_size = int.from_bytes(file[14:18], byteorder="little", signed=True)
        
        if header_size != 40:
            raise Exception("Unsupported BMP format, only 1, 2, 4, 8, 24 or 32-bit BMP with standard header is supported.")

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

    def _loadPalette(self) -> np.array:
        pass

    def loadColorData(self) -> np.array:
        pass

    def convertToBMP(self) -> bytearray:
        pass

    def saveBMP(self) -> bool:
        pass

bmp = BMPImage()
print(bmp.loadBMP(path="./media/bmp/24bit.bmp"))
