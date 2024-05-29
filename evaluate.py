import argparse
import warnings
warnings.filterwarnings("ignore")
from meshseg.evaluation.eval_functions import *

if __name__ == "__main__":
    # Get the config file path from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", help="path to the output dir.", default="outputs/FAUST/head")
    parser.add_argument("-mesh_name", help="eval mesh name. such as tr_scan_000.obj", default="input/FAUST/mesh_name.txt")
    parser.add_argument("-fine_grained", help="fine grained evaluation", default=False)
    args = parser.parse_args()
    print(args)

    evaluate_faust(args)