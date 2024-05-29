import os
import math
import torch
from ast import literal_eval


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def prepare_seg_classes(input_prompt):
    prompt = str(input_prompt)
    ps = literal_eval(prompt)
    part_names = sorted(ps)

    ind = 0
    cls_name_to_id = {}
    cls_id_to_name = {}

    for i, p in enumerate(part_names):
        cls_name_to_id[p] = ind
        cls_id_to_name[ind] = p
        ind += 1
    cls_name_to_id["unknown"] = ind
    cls_id_to_name[ind] = "unknown"
    print(cls_name_to_id)

    return part_names, cls_name_to_id, cls_id_to_name

def get_elev_azim(num_views):
    n = 360 / num_views
    angles = [i * n for i in range(num_views)]
    weidu = [angle / 180 * math.pi for angle in angles]
    jingdu = [0.4] * num_views
    return torch.tensor(jingdu), torch.tensor(weidu)

def save_pos(elev, azim, r, save):
    a = elev.numpy()
    b = azim.numpy()
    c = r
    result = [[x, y, c] for x, y in zip(a, b)]
    with open(save, 'w') as file:
        for sublist in result:
            file.write(' '.join(map(str, sublist)) + '\n')
    print("save pos success!")


def get_view(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    jingdu = []
    weidu = []
    for value in lines:
        values = value.split()
        jingdu_value = float(values[0])
        weidu_value = float(values[1])
        jingdu.append(jingdu_value)
        weidu.append(weidu_value)
    jingdu = torch.tensor(jingdu)
    weidu = torch.tensor(weidu)

    return jingdu, weidu


colors_lst = [
    [247 / 255, 165 / 255, 94 / 255.0],
    [247 / 255.0, 94 / 255.0, 165 / 255.0],
    [94 / 255.0, 165 / 255.0, 247 / 255.0],
    [100 / 255.0, 0, 100 / 255.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.5, 1.0],
    [0.2, 0.2, 0.9],
    [0.1, 0.5, 0.9],
    [0.9, 0.2, 0.9],
    [0.9, 0.4, 0.1],
    [0.5, 0.0, 0.9],
    [0.1, 0.9, 0.5],
]