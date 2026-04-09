from amb3r.training_stage2v2 import get_args_parser, main

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
