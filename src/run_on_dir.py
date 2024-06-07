import os
import argparse
from run_all import run_on_sketch


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", type=str, required=True)
    args = argparser.parse_args()

    sketches = os.listdir(args.dir)
    for sketch in sketches:
        run_on_sketch(os.path.join(args.dir, sketch))


if __name__ == "__main__":
    main()
