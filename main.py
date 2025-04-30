from content.src.parser import create_parser
from content.src.handle import handle_generate_wav, handle_generate_bmp, handle_conversion, handle_info
from content.src.convertor import Convertor


def main():
    parser = create_parser()
    args = parser.parse_args()
    convertor = Convertor()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.generate_wav and args.generate_bmp:
        parser.error("Cannot use both --generate-wav and --generate-bmp at the same time.")

    if args.convert and args.info:
        parser.error("Cannot use both --convert and --info at the same time.")


    if args.generate_wav:
        handle_generate_wav(args, convertor)

    elif args.generate_bmp:
        handle_generate_bmp(args, convertor)

    elif args.convert:
        handle_conversion(args, convertor)

    elif args.info:
        handle_info(args, convertor)

    if args.verbose:
        print("Verbose mode enabled")
        print("All arguments:", args)

if __name__ == "__main__":
    main()

    # Example usage:
    # Generate a random BMP image with dimensions 100x100 and save to output.bmp
    # python parser.py --generate-bmp 100 100 output.bmp

    # Generate a random WAV file with 44100 samples and save to output.wav
    # python parser.py --generate-wav 44100 output.wav

    # Generate a sine wave WAV file with 44100 samples at 440Hz and save to output.wav
    # python parser.py --generate-wav 44100 output.wav --freq 440

    # Convert a BMP file to a WAV file
    # python parser.py --convert input.bmp output.wav

    # Convert a WAV file to a BMP file with dimensions 100x100
    # python parser.py --convert input.wav output.bmp --dimensions 100 100

    # Display metadata about a BMP file
    # python parser.py --info input.bmp

    # Display metadata about a WAV file
    # python parser.py --info input.wav

    # Enable verbose mode for detailed output
    # python parser.py --generate-bmp 100 100 output.bmp --verbose

    # Show plots in matplotlib without axes
    # python parser.py --info input.bmp --show

    # Show plots in matplotlib with axes
    # python parser.py --info input.bmp --show-with-axes