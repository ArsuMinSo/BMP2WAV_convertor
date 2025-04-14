import argparse
from convertor import Convertor
import matplotlib.pyplot as plt



def display_bmp_and_wav(bmp_array, wav_data):
    fig, ax = plt.subplots(nrows=2, figsize=(10, 6))

    ax[0].imshow(bmp_array[::-1])  # BMP image
    ax[0].set_title("Input BMP Image")
    ax[0].axis('off')

    ax[1].plot(wav_data, linewidth=0.5)  # WAV waveform
    ax[1].set_title("Generated WAV Waveform")

    plt.tight_layout()
    plt.show()


def display_wav_and_bmp(wav_data, bmp_array):
    fig, ax = plt.subplots(nrows=2, figsize=(10, 6))

    ax[0].plot(wav_data, linewidth=0.5)  # WAV waveform
    ax[0].set_title("Input WAV Waveform")

    ax[1].imshow(bmp_array[::-1])  # Reconstructed BMP
    ax[1].set_title("Reconstructed BMP Image")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog='BMP ↔ WAV Convertor',
        description='Converts BMP images to WAV files and vice versa, or generates random images.',
        epilog='Made with ❤️'
    )

    parser.add_argument('-g', '--generate',
                        nargs=3, metavar=('WIDTH', 'HEIGHT', 'OUTFILE'),
                        help='Generate random BMP image.')

    parser.add_argument('-c', '--convert',
                        nargs=2, metavar=('INPUTFILE', 'OUTPUTFILE'),
                        help='Convert one format to another.')

    parser.add_argument('-v', '--verbose',
                        action='store_true', help='Verbose mode.')

    parser.add_argument('--show',
                        action='store_true', help='Show plots in matplotlib.')

    args = parser.parse_args()
    convertor = Convertor()

    if args.generate:
        width, height, outfile = args.generate
        width = int(width)
        height = int(height)
        result = convertor.generateRandomPic(width, height, outfile)
        print(f"Generated image saved to {result}")

    if args.convert:
        input_file, output_file = args.convert
        
        if input_file.endswith(".bmp") and output_file.endswith(".wav"):   
            if args.verbose:
                print(f"Reading BMP from {input_file}")
            
            bmp_array = convertor.bmp.loadBMP(input_file)
            
            if args.verbose:
                print(f"Saving WAV to {output_file}")
            
            wav_data = convertor.bmp2wav(input_file, output_file)
            
            print(f"Converted BMP to WAV: {output_file}")
            if args.show:
                convertor.display_image(bmp_array[2], wav_data[1])
        
        elif input_file.endswith(".wav") and output_file.endswith(".bmp"):
            if args.verbose: print(f"Reading WAV from {output_file}")
            wav_data = convertor.wav.load_wav(output_file)
            if args.verbose: print(f"Saving BMP to {input_file}")
            bmp_array = convertor.wav2bmp(output_file, input_file)
            print(f"Converted WAV to BMP: {input_file}")
            if args.show:
                convertor.display_wav(wav_data, bmp_array[1])
        else:
            raise ValueError("Wrong files")
        

    if args.verbose:
        print("📢 Verbose mode enabled")
        print("All arguments:", args)


if __name__ == '__main__':
    main()
