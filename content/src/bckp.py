
class Convertor:
    """
    Class for converting BMP images into a NumPy array representation.
    Supports 1, 4, 8, 24 and 32-bit BMP formats.
    """


    def __init__(self, file: str):
        self.binary_file = open(file, "rb").read()
        if file.endswith(".bmp"):
            self._isaudio = False
            self.load_bmp()
        elif file.endswith(".wav"):
            self._isaudio = True
            self.load_wav()

    def load_bmp(self):
        """
        Loads BMP file, extracts the palette, and converts to NumPy array.
        """
        self._info = None
        self.palette_info = None
        self.pixel_info = None
        self.extract_bmp_header()
        self.extract_palette()
        self.convert_bmp_to_numpy()
    
    def extract_bmp_header(self):
        """
        Reads and extracts header information from the BMP file.
        Only supports standard BMP headers of size 40 bytes.
        """
        header_size = int.from_bytes(self.binary_file[14:18], byteorder="little", signed=True)
        
        if header_size != 40:
            raise Exception("Unsupported BMP format, only 1, 2, 4, 8, 24 or 32-bit BMP with standard header is supported.")

        self._info = {
            "BMP identifier": self.binary_file[:2].decode("utf-8"),
            "File size": int.from_bytes(self.binary_file[2:6], byteorder="little", signed=True),
            "Reserved bytes": int.from_bytes(self.binary_file[6:10], byteorder="little", signed=True),
            "Palette offset": int.from_bytes(self.binary_file[10:14], byteorder="little", signed=True),
            "Header size": int.from_bytes(self.binary_file[14:18], byteorder="little", signed=True),
            "Width": int.from_bytes(self.binary_file[18:22], byteorder="little", signed=True),
            "Height": int.from_bytes(self.binary_file[22:26], byteorder="little", signed=True),
            "Planes": int.from_bytes(self.binary_file[26:28], byteorder="little", signed=True),
            "Bits per pixel": int.from_bytes(self.binary_file[28:30], byteorder="little", signed=True),
            "Compression": int.from_bytes(self.binary_file[30:34], byteorder="little", signed=True),
            "Image size": int.from_bytes(self.binary_file[34:38], byteorder="little", signed=True),
            "Horizontal resolution": round(int.from_bytes(self.binary_file[38:42], byteorder="little", signed=True)/100 *2.54, 0),
            "Vertical resolution": round(int.from_bytes(self.binary_file[42:46], byteorder="little", signed=True)/100 *2.54, 0),
            "Number of colors in the palette": int.from_bytes(self.binary_file[46:50], byteorder="little", signed=True),
            "Important colors": int.from_bytes(self.binary_file[50:54], byteorder="little", signed=True),
        }

    def extract_palette(self):
        """
        Extracts the color palette from BMP file if applicable (for 1, 4, and 8-bit BMPs).
        """
        if self._info["Bits per pixel"] in [1, 4, 8]:
            palette_hex = (self.binary_file[54:self._info["Palette offset"]])

            num_colors = self._info["Number of colors in the palette"] or 2 ** self._info["Bits per pixel"]
            self.palette_info = np.zeros([num_colors, 4], dtype=np.uint8)

            for i in range(0, len(palette_hex), 4):
                B, G, R, U  = palette_hex[i], palette_hex[i+1], palette_hex[i+2], palette_hex[i+3]
                self.palette_info[i//4] = [R, G, B, 255 - U]
        else:
            self.palette_info = None

    def convert_bmp_to_numpy(self):
        """Converts BMP pixel data into a NumPy array."""
        pixel_data = self.binary_file[self._info["Palette offset"]:]
        width, height = self._info["Width"], self._info["Height"]
        bpp = self._info["Bits per pixel"]

        scanline_size = self.get_scanline_size(width, bpp)

        if bpp in [1, 4, 8]:
            self.pixel_info = self.decode_indexed_image(pixel_data, width, height, bpp, scanline_size)
        elif bpp in [24, 32]:
            self.pixel_info = self.decode_direct_image(pixel_data, width, height, bpp, scanline_size)

    def get_scanline_size(self, width, bpp):
        """Calculates the BMP scanline size (padded to 4-byte boundaries)."""
        return (math.ceil((width * bpp) / 32) * 32) // 8

    def decode_indexed_image(self, pixel_data, width, height, bpp, scanline_size):
        """Decodes BMP images with a color palette (1, 4, 8-bit) using scanlines."""
        pixel_plane = np.zeros([height, width, 4], dtype=np.uint8)
        
        byte_index = 0
        for y in range(height):
            row_pixels = self.decode_indexed_row(pixel_data[byte_index:byte_index + scanline_size], width, bpp)
            pixel_plane[height - 1 - y] = row_pixels
            byte_index += scanline_size  # Move to next scanline
        return pixel_plane

    def decode_indexed_row(self, row_data, width, bpp):
        """Decodes a single scanline for indexed BMP images."""
        row = np.zeros([width, 4], dtype=np.uint8)
        byte_index = 0
        bit_offset = 0

        for x in range(width):
            if bpp == 8:
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

            row[x] = self.palette_info[index]

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

    def load_wav(self):
        """
        Loads WAV file, extracts the header and footer, and converts to NumPy array.
        """
        self._info = None
        self.sound_bytes = None

        self.extract_wav_header()
        self.convert_wav_to_numpy()

    def convert(self, output_file):
        with open(output_file, "wb") as out_file:
            if not self._isaudio:
                out_file.write(self.convert_bmp2wav())
            else:
                out_file.write(self.convert_wav2bmp())

    def extract_wav_header(self):
        """
        Reads and extracts header information from the WAV file.
        Only supports standard WAV headers of size 40 bytes.
        """

        self._info = {
            "RIFF Header": self.binary_file[:4].decode("utf-8"),
            "File size": int.from_bytes(self.binary_file[4:8], byteorder="little", signed=True),
            "WAVE": self.binary_file[8:12].decode("utf-8"),

            "Signature 'fmt '": self.binary_file[12:16].decode("utf-8"),
            "Header size": int.from_bytes(self.binary_file[16:20], byteorder="little", signed=True),
            "Format Tag": int.from_bytes(self.binary_file[20:22], byteorder="little", signed=True),
            "Channels": int.from_bytes(self.binary_file[22:24], byteorder="little", signed=True),
            "Sample Rate": int.from_bytes(self.binary_file[24:28], byteorder="little", signed=True),
            "Bytes per second": int.from_bytes(self.binary_file[28:32], byteorder="little", signed=True),
            "Block size": int.from_bytes(self.binary_file[32:34], byteorder="little", signed=True),
            "Bits per sample": int.from_bytes(self.binary_file[34:36], byteorder="little", signed=True),
            "'Data'": self.binary_file[36:40].decode("utf-8"),
            "Length": int.from_bytes(self.binary_file[40:44], byteorder="little", signed=True),
            
            "Width BMP": int.from_bytes(self.binary_file[-8:-4],byteorder="little")
                    if self.binary_file.find(b"Edat") != -1
                    else ceil(sqrt(int.from_bytes(self.binary_file[40:44], byteorder="little", signed=True))),
            "Height BMP": int.from_bytes(self.binary_file[-4:],byteorder="little")
                    if self.binary_file.find(b"Edat") != -1
                    else ceil(sqrt(int.from_bytes(self.binary_file[40:44], byteorder="little", signed=True))),
        }
        
    def convert_wav_to_numpy(self):
        if self._info["Channels"] == 1 and self._info["Sample Rate"] == 44100 and self._info["File size"] - 36 >= self._info["Length"]:
            self.sound_bytes = np.zeros([self._info["Length"]//(self._info["Bits per sample"]//8)])
            for byte_index in range(44, 44+self._info["Length"], self._info["Bits per sample"]//8):
                self.sound_bytes[(byte_index - 44)//4] = int.from_bytes(self.binary_file[byte_index:byte_index+self._info["Bits per sample"]//8], byteorder="little", signed=True)
            return self.sound_bytes
        
        print(f"chan: {self._info["Channels"]} sample: {self._info["Sample Rate"]} size{self._info["File size"] - 36, self._info["Length"]}")
        raise Exception("Unsuported WAV format")


    def convert_bmpnp_to_wawnp(self):
        height, width, depth = self.pixel_info.shape

        data = np.zeros([width * height])

        for y in range(height):
            for x in range(width):
                R, G, B, A = self.pixel_info[y, x]
                #print(f"{x, y}: {R, G, B, A, R<<23 or G<<15 or B<<7 or A}")
                data [y * width + x] =  int(((R)<<24) + ((G)<<16) + ((B)<<8) + A - (2**31-1))
                if data[y * width + x] > (2**31 -1): data[y * width + x] = (2**31 -1)
                elif data[y * width + x] < (-2**31): data[y * width + x] = (-2**31)
                else: pass
        return data
    """
    
    def convert_bmpnp_to_wawnp(self):
        
        height, width, depth = self.pixel_info.shape

        raw_audio_data = self.sound_bytes

        pixel_data = np.zeros([width, height, 4])

        for i in range(width * height):
                sample = int.from_bytes(raw_audio_data[i * 4:(i + 1) * 4], "little", signed=True)

                # Extract RGBA values
                a = sample & 0xFF
                b = (sample >> 8) & 0xFF
                g = (sample >> 16) & 0xFF
                r = (sample >> 24) & 0xFF

                pixel_data[height//i, width%i] = ((r, g, b, a))
        return pixel_data
        """    

    def convert_bmp2wav(self):
        """Converts BMP to WAV, creates header and adds sound data"""

        self.load_bmp()
        self.convert_bmp_to_numpy()

        samples = self.convert_bmpnp_to_wawnp()

        data = b""

        height, width, depth = self.pixel_info.shape

        for sample in samples:
            
            data += int(sample).to_bytes(length= 4, byteorder="little", signed=True)

        sample_rate = 44100  # Hz
        num_channels = 1  # Mono
        bits_per_sample = 32  
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)

        # WAV hlavička (ručně sestavená)
        header = (
            b'RIFF' +                       # Chunk ID
            (36 + len(data)).to_bytes(4, 'little') +  # Chunk size
            b'WAVE' +                       # Format
            b'fmt ' +                       # Subchunk1 ID
            (16).to_bytes(4, 'little') +    # Subchunk1 size (PCM)
            (1).to_bytes(2, 'little') +     # Audio format (1 = PCM)
            num_channels.to_bytes(2, 'little') +  # Number of channels
            sample_rate.to_bytes(4, 'little') +   # Sample rate
            byte_rate.to_bytes(4, 'little') +     # Byte rate
            block_align.to_bytes(2, 'little') +   # Block align
            bits_per_sample.to_bytes(2, 'little') +  # Bits per sample
            b'data' +                       # Subchunk2 ID
            len(data).to_bytes(4, 'little')  # Subchunk2 size
        )

        footer = (
            b"Edat" +
            (width).to_bytes(4, 'little') +
            (height).to_bytes(4, 'little')    
        )

        return header + data + footer  # Vrátí hlavičku a data
        
    def convert_wav2bmp(self):
        """Converts WAV to BMP manually without using external libraries."""

        self.load_wav()  # Load binary WAV data into self.binary_file

        width, height = self._info["Width BMP"], self._info["Height BMP"]
        # Extract width and height from the last 8 bytes (Little Endian format)
        pixel_data = self.convert_wawnp_to_bmpnp()

        samples = pixel_data.shape[0] * pixel_data.shape[1]

        # Extract raw audio data (skip WAV header and footer)

        if samples < width * height:
            raise ValueError("Not enough samples to reconstruct the image.")

        # Convert samples back to pixels

        # Create BMP pixel array (bottom-up order)
        bmp_data = b""
        row_padding = (4 - (width * 4) % 4) % 4  # BMP rows must be aligned to 4-byte boundaries

        for y in range(height - 1, -1, -1):  # BMP stores rows from bottom to top
            row_data = b"".join(
                pixel_data[(y * width) + x][2].to_bytes(1, "little") +  # Blue
                pixel_data[(y * width) + x][1].to_bytes(1, "little") +  # Green
                pixel_data[(y * width) + x][0].to_bytes(1, "little") +  # Red
                pixel_data[(y * width) + x][3].to_bytes(1, "little")    # Alpha
                for x in range(width)
            )
            bmp_data += row_data + (b"\x00" * row_padding)  # Add row padding if necessary

        # Create BMP Header (54 bytes)
        file_size = 54 + len(bmp_data)  # BMP file size
        bmp_header = (
            b'BM' +                         # BMP Signature
            file_size.to_bytes(4, 'little') +  # File Size
            b'\x00\x00' + b'\x00\x00' +     # Reserved
            (54).to_bytes(4, 'little') +    # Pixel Data Offset
            (40).to_bytes(4, 'little') +    # Header Size
            width.to_bytes(4, 'little') +   # Image Width
            height.to_bytes(4, 'little') +  # Image Height
            (1).to_bytes(2, 'little') +     # Color Planes
            (32).to_bytes(2, 'little') +    # Bits Per Pixel (RGBA)
            (0).to_bytes(4, 'little') +     # Compression (None)
            len(bmp_data).to_bytes(4, 'little') +  # Image Size
            (2835).to_bytes(4, 'little') +  # X Pixels Per Meter
            (2835).to_bytes(4, 'little') +  # Y Pixels Per Meter
            (0).to_bytes(4, 'little') +     # Colors Used
            (0).to_bytes(4, 'little')       # Important Colors
        )

        return bmp_header + bmp_data  # Return BMP file content
    
    def display_image(self):
        """Displays the decoded BMP image using Matplotlib."""
        
        fig, ax = plt.subplots(nrows=2)

        ax[0].imshow(self.pixel_info[::-1])
        ax[0].set_ylim(-10, self.pixel_info.shape[0] + 10)
        ax[0].set_xlim(-10, self.pixel_info.shape[1] + 10)
        ax[0].set_title("Decoded BMP Image")

        data = self.convert_bmpnp_to_wawnp()
        print(max(data), min(data))
        ax[1].plot(np.arange(0, data.shape[0], dtype=int), data)

        plt.show()

    def display_wav(self):
        fig, ax = plt.subplots(nrows=2)

        data = self.sound_bytes
        print(max(data), min(data))
        ax[0].set_title("Decoded WAV wave")
        ax[0].plot(np.arange(0, data.shape[0], dtype=int), data)

        ax[1].imshow(self.pixel_info[::-1])
        ax[1].set_ylim(-10, self.pixel_info.shape[0] + 10)
        ax[1].set_xlim(-10, self.pixel_info.shape[1] + 10)
        ax[1].set_title("Reconstruced BMP image")
        
        plt.show()



"""
#cnv = Convertor("./src/wav/tone1000Hz44100Hz1A.wav")
#cnv = Convertor("./out/out1.wav")
#dataframe = pd.DataFrame.from_dict(bmp._info, orient="index", columns=["Value"])
bmp = Convertor("./src/bmp/01.bmp")
bmp.convert("./out/out1.wav")
bmp.display_image()

wav = Convertor("./out/out1.wav")
bmp.convert("./out/out1.wav")
wav.display_wav()
#print(cnv.pixel_info)

#print(cnv.sound_bytes)
#print(cnv)


"""