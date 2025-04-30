import argparse
from content.src.wav import DEFAULT_FREQ



def create_parser() -> argparse.ArgumentParser:
    """Creates and returns the argument parser."""
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
                        help='Generate random WAV wave. If ISSINE is true, generates sine wave.')

    parser.add_argument('-f', '--freq',
                        type=int,
                        metavar='FREQENCY',
                        default=None,
                        help=f'Frequency of the sine wave - used with generate-wav (default: {DEFAULT_FREQ}Hz).')

    parser.add_argument('-c', '--convert',
                        nargs=2,
                        metavar=('INFILE', 'OUTFILE'),
                        help='Convert one format to another.')

    parser.add_argument('-d', '--dimensions',
                        type=int,
                        nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Dimensions of picture for WAV to BMP conversion.')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose mode.')

    parser.add_argument('-i', '--info',
                        metavar='FILE',
                        help='Shows metadata about file.')

    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='Show plots in matplotlib without axes.')

    parser.add_argument('-sx', '--show-with-axes',
                        action='store_true',
                        help='Show axes in matplotlib.')
    
    parser.add_argument('--test',
                        action='store_true',
                        help='Save plots instead of displaying them (for testing).')

    return parser
