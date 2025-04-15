import argparse
from convertor import Convertor


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
                        help='Generate random WAV wave.')

    parser.add_argument('-c', '--convert',
                        nargs=2,
                        metavar=('INFILE', 'OUTFILE'),
                        help='Convert one format to another.')

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
        result = convertor.generateRandomWav(samples, outfile)
        if args.verbose:
            print(f"Generated wav saved to {result}")
        if args.show:
            wav_data = convertor.wav.load_wav(result)
            convertor.wav.display_wav(wav_data[1])


    elif args.generate_bmp:
        width, height, outfile = args.generate_bmp
        width = int(width)
        height = int(height)
        result = convertor.generateRandomPic(width, height, outfile)
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
            if args.verbose:
                print(f"Reading WAV from {input_file}")

            wav_data = convertor.wav.load_wav(input_file)
            
            if args.verbose:
                print(f"Saving BMP to {output_file}")

            bmp_array = convertor.wav2bmp(input_file, output_file)
            
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

    # -c .\content\media\bmp\01.bmp .\content\media\out\out.wav --show

if __name__ == '__main__':
    main()

