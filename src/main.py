import pandas as pd
import numpy as np

from bmp import BMPFormat
from wav import WAVFormat

bmp = BMPFormat()
wav = WAVFormat()


header, sound = wav.load_wav(path="./media/wav/u8bit.wav")
dataFrame = pd.DataFrame.from_dict(header, orient="Index", columns=["Value"])
print(dataFrame, end= "\n\n")
print(sound.shape, end= "\n\n")
wav.display_wav(sound_data=sound)

"""path = bmp.saveBMP(pixel_array=np.random.randint(low=0, high=256, size=[300, 100, 4], dtype="uint8"), path= "./media/out/24bitOut.bmp")
header, palette, pixel_plane = bmp.loadBMP(path)

dataFrame = pd.DataFrame.from_dict(header, orient="Index", columns=["Value"])
print(dataFrame, end= "\n\n")
print(palette, end= "\n\n")
print(pixel_plane.shape, end= "\n\n")
bmp.display_image(pixel_array=pixel_plane, wav_data= None)
"""
