from content.src.convertor import Convertor
from content.src.wav import DEFAULT_FREQ
from content.src.bmp import DEFAULT_DIMENSIONS


def handle_generate_wav(args, convertor: Convertor):
    """Handles the generation of WAV files."""
    samples = int(args.generate_wav[0])  # Počet vzorků
    outfile = args.generate_wav[1]       # Výstupní soubor
    if args.freq is None:
        result = convertor.wav.generateRandomWav(samples, outfile)
    else:
        result = convertor.wav.generateSinWav(samples, outfile, args.freq)

    if args.verbose:
        print(f"Generated WAV file saved to {result}")

    if args.show:
        wav_data = convertor.wav.load_wav(result) 
        convertor.wav.display_wav(wav_data[1], show_axes=args.show_with_axes, test_mode=args.test)

def handle_generate_bmp(args, convertor: Convertor):
    """Handles the generation of BMP files."""
    width, height = map(int, args.generate_bmp[:2])
    outfile = args.generate_bmp[2]
    result = convertor.bmp.generateRandomPic(width, height, outfile)

    if args.verbose:
        print(f"Generated BMP file saved to {result}")

    if args.show or args.show_with_axes:
        pixel_data = convertor.bmp.load_bmp(result)
        convertor.bmp.display_image(pixel_data[2], show_axes=args.show_with_axes, test_mode=args.test)


def handle_conversion(args, convertor: Convertor):
    """Handles file format conversion."""
    input_file, output_file = args.convert

    if input_file.endswith(".bmp") and output_file.endswith(".wav"):
        if args.verbose:
            print(f"Converting BMP to WAV: {input_file} → {output_file}")
        bmp_array = convertor.bmp.load_bmp(input_file)
        convertor.bmp2wav(input_file, output_file)

        if args.show:
            wav_data = convertor.wav.load_wav(output_file)
            convertor.bmp.display_image(bmp_array[2], wav_data[1], show_axes=args.show_with_axes, test_mode=args.test)

    elif input_file.endswith(".wav") and output_file.endswith(".bmp"):
        if not args.dimensions:
            args.dimensions = DEFAULT_DIMENSIONS
        width, height = args.dimensions
        if args.verbose:
            print(f"Converting WAV to BMP: {input_file} → {output_file}")
        wav_data = convertor.wav.load_wav(input_file)
        convertor.wav2bmp(input_file, output_file, width, height)

        if args.show:
            bmp_array = convertor.bmp.load_bmp(output_file)
            convertor.wav.display_wav(wav_data[1], bmp_array[2], show_axes=args.show_with_axes, test_mode=args.test)
    else:
        raise ValueError("Invalid file extensions for conversion.")

def handle_info(args, convertor: Convertor):
    """Handles displaying metadata about a file."""
    input_file = args.info

    if input_file.endswith(".bmp"):
        if args.verbose:
            print(f"Reading BMP metadata from {input_file}")
        bmp_array = convertor.bmp.load_bmp(input_file)
        convertor.bmp.print_file_info(bmp_array[0])

        if args.show:
            convertor.bmp.display_image(bmp_array[2], show_axes=args.show_with_axes, test_mode=args.test)

    elif input_file.endswith(".wav"):
        if args.verbose:
            print(f"Reading WAV metadata from {input_file}")
        wav_data = convertor.wav.load_wav(input_file)
        convertor.wav.print_file_info(wav_data[0])

        if args.show:
            convertor.wav.display_wav(wav_data[1], show_axes=args.show_with_axes, test_mode=args.test)
    else:
        raise ValueError("Unsupported file type for metadata.")
