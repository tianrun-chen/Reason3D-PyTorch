import argparse
from Inference_functions import SegmentNet

if __name__ == "__main__":
    # Get the config file path from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_root", help="input data path.", default="data")
    parser.add_argument("-mesh_name", help="input mesh data name.", default="cage.obj")
    parser.add_argument("-output_dir", help="output dir path.", default="outputs")
    parser.add_argument("-pose_file", help="mesh face attributes color", default="data/pose.txt")
    parser.add_argument("-step2_data", help="run step3 use step2 data path.", default="outputs/step_2")
    # "prompt": If it is to inference FAUST data, you need to select one of ['arm', 'head', 'leg', 'torso']or fine_grained; other data do not need to be modified.
    parser.add_argument("-input_prompt", help="input prompt.", default="lable")
    parser.add_argument("-step", help="to run which step.", default="step_3")
    parser.add_argument("-gpu", help="gpu id", default=0, type=int)

    parser.add_argument("-color", help="mesh face attributes color", default=None)
    parser.add_argument("-image_size", help="render image size", type=int, default=1024)
    parser.add_argument("-view_num", help="render image number", type=int, default=8)
    parser.add_argument("-frontview_std", help="mesh zoom in and zoom out", default=8)
    parser.add_argument("-frontview_center", help="center elev and center elev", default=[3.14, 0.0])

    parser.add_argument("-face_smoothing_n_ring", help="Number of smoothing neighbor faces", default=5)
    parser.add_argument("-face_smoothing", help="use or not smoothing", default=True)
    parser.add_argument("-gaussian_reweighting", help="use or not gaussian", default=True)

    args = parser.parse_args()
    print(args)

    SegmentNet(
        opt=args
    )
