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
