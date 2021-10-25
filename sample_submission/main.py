from Predictor import Predictor
from argparse import ArgumentParser

def main(input_dir, output_file):
    predictor = Predictor()
    predictor.main(input_dir, output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-input", type=str, required=True,
        help="Input directory of test videos"
    )
    parser.add_argument(
        "-output", type=str, required=True,
        help="Output dir with filename e.g. /data/submission.csv"
    )

    args = parser.parse_args()
    main(input_dir=args.input, output_file=args.output)
