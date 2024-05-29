import os
import json
import trimesh
import torch
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial import distance_matrix
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist

from meshseg.mesh import Mesh, MeshNormalizer


def calculate_shape_IoU(pred: np.array, gt: np.array, label: str):
    # pred: np.array [N_points]
    # seg: np.array [N_points]
    # Adopted from https://github.com/antao97/dgcnn.pytorch/blob/f4da503444ce663b06c8d1bca79e746ef1647b18/main_partseg.py
    I = np.sum(np.logical_and(pred == label, gt == label))
    # print(f"right: {I}")
    U = np.sum(np.logical_or(pred == label, gt == label))
    # print(f"right + error: {U}")
    if U == 0:
        iou = 1.0
    else:
        iou = I / float(U)
    return iou


def evaluate_faust(opt): # dataset_output_dir, fine_grained=False
    # Read the mesh list
    with open(opt.mesh_name) as fin:
        meshes = [el.strip() for el in fin.readlines() if len(el) > 0 and "obj" in el]

    # Read the gt
    if opt.fine_grained:
        print("Fine grained")
        with open("input/FAUST/fine_grained_gt.json") as fin:
            dataset_gt = json.load(fin)
    else:
        print("coarse")
        with open("input/FAUST/coarse_gt.json") as fin:
            dataset_gt = json.load(fin)

    # Start computing the mIoUs
    for m in meshes:
        print(m)
        partsIous = defaultdict(list)
        mesh_name = m.split(".")[0]
        output_dir = os.path.join(opt.dataset_output_dir, mesh_name)

        gt = dataset_gt[mesh_name]
        gtLabels = sorted(list(set(gt)))
        gt = np.array(gt)

        if os.path.isfile(os.path.join(output_dir, "step_3", "vert_preds.json")):
            p = os.path.join(output_dir, "step_3", "vert_preds.json")
        else:
            raise Exception(f"File {p} does not exist")

        with open(p) as fin:
            pred_list = [t.replace("_", " ") for t in json.load(fin)]
            pred = np.array(pred_list)

        for l in gtLabels:
            partsIous[l].append(calculate_shape_IoU(pred, gt, l))
        print(output_dir)
        mIous = []
        # save = os.path.join(output_dir, f"step_3/Miou_fine_grained.txt")
        save = os.path.join(output_dir, "Miou_coarse.txt")
        with open(save, 'w') as file:
            for k, v in partsIous.items():
                mIous.append(np.mean(v))
                file.write(f"{k}:{np.mean(v)}\n")
                print(f"{k}:{np.mean(v)}")

            print(f"Experiment {opt.dataset_output_dir}: {np.mean(mIous)}")

            print(f"save : {save}")
            file.write(f"all:{np.mean(mIous)}\n")
