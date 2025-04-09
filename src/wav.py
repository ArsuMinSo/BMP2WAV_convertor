from matplotlib import pyplot as plt
import numpy as np
from numpy import sqrt, ceil

class WAVFormat:
    def __init__(self):
        pass


    def load_wav(self, path:str) -> tuple:
        """
        Loads WAV file, extracts the header and footer, and converts to NumPy array.
        """
        if path.endswith(".wav"):
            with open(path, "rb") as file:
                file = file.read()
                wav_header = self._loadHeader(file)
                sound_data = self.decode_wav(wav_header, file)
                
        return (wav_header, sound_data)

    def _loadHeader(self, file):
        """
        Reads and extracts header information from the WAV file.
        Only supports standard WAV headers of size 40 bytes.
        """

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
            
            "Width BMP": int.from_bytes(file[-8:-4],byteorder="little")
                    if file.find(b"Edat") != -1
                    else ceil(sqrt(int.from_bytes(file[40:44], byteorder="little", signed=True))),
            "Height BMP": int.from_bytes(file[-4:],byteorder="little")
                    if file.find(b"Edat") != -1
                    else ceil(sqrt(int.from_bytes(file[40:44], byteorder="little", signed=True))),
        }
        return header

    def decode_wav(self, header, file):
        # Získání informací o souboru
        sample_width = header["Bits per sample"] // 8
        num_samples = header["Length"] // sample_width
        num_channels = header["Channels"]

        datatype = np.uint8 if header["Bits per sample"] == 8 else np.int16 if header["Bits per sample"] == 16 else np.int32

        # Předalokování pole pro zvuková data
        sound_bytes = np.zeros([num_samples // num_channels, num_channels], dtype=datatype)

        # Načítání vzorků pro každý kanál
        for i in range(0, header["Length"], sample_width * num_channels):
            for ch in range(num_channels):
                byte_index = 44 + i + ch * sample_width
                sample = int.from_bytes(
                    file[byte_index:byte_index + sample_width],
                    byteorder="little", signed=True
                )
                sound_bytes[i // (sample_width * num_channels), ch] = sample

        return sound_bytes
    
    def display_wav(self, sound_data, pixel_info = None):
        fig, ax = plt.subplots(nrows=2)

        ax[0].set_title("Decoded WAV wave")
        ax[0].plot(np.arange(0, sound_data.shape[0], dtype=int), sound_data)

        if pixel_info:
            ax[1].imshow(self.pixel_info[::-1])
            ax[1].set_ylim(-10, self.pixel_info.shape[0] + 10)
            ax[1].set_xlim(-10, self.pixel_info.shape[1] + 10)
            ax[1].set_title("Reconstruced BMP image")
        
        plt.show()
