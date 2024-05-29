import os
import json
import torch

import kaolin.ops.mesh
import os.path as osp

from meshseg.mesh import MeshNormalizer
from meshseg.mesh import Mesh
from meshseg.renderer.renderer import Renderer
from meshseg.methods.segmentors import *

from utils.mesh_utils import ensure_directory, prepare_seg_classes, get_elev_azim, save_pos, get_view,  colors_lst


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def SegmentNet(opt):
    # Get mesh path
    mesh_path = os.path.join(opt.data_root, opt.mesh_name)
    print(mesh_path)
    assert os.path.isfile(mesh_path)

    ensure_directory(opt.output_dir)

    # Save the config in the output folder
    opt_dict = vars(opt)
    with open(os.path.join(opt.output_dir, 'config.json'), 'w') as json_file:
        json.dump(opt_dict, json_file, indent=4)

    # Convert label to id
    part_names, cls_name_to_id, cls_id_to_name = prepare_seg_classes([opt.input_prompt])

    # Read the mesh
    print(f"Reading the mesh...")
    mesh = Mesh(mesh_path)
    print(
        f"Reading the mesh : {opt.mesh_name} having {mesh.faces.shape[0]} faces and {mesh.vertices.shape[0]} vertices"
    )

    # Normalize mesh in unit sphere
    _ = MeshNormalizer(mesh)()

    # Put default grey color to the mesh
    if opt.color == None:
        mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
            torch.ones(1, len(mesh.vertices), 3).to(device)
            * torch.tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device),
            mesh.faces,
        )
    else:
        mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
            torch.ones(1, len(mesh.vertices), 3).cuda()
            * torch.tensor(opt.color).unsqueeze(0).unsqueeze(0).cuda(),
            mesh.faces,
        )

    # Create the renderer
    print(f"Creating the renderer...")
    render = Renderer(dim=(opt.image_size, opt.image_size))

    # Initialize Background
    background = torch.tensor([0.0, 0.0, 0.0]).to(device)

    if opt.step == "step_1":
        print("The code is currently running: Render Image")
        ensure_directory(os.path.join(opt.output_dir, 'step_1'))
        with torch.no_grad():
            print(f"Rendering the views...")
            if opt.pose_file == None:
                elev, azim = get_elev_azim(opt.view_num)
            else:
                elev, azim = get_view(opt.pose_file)
            save_path = os.path.join(opt.output_dir, 'step_1', "view.txt")
            rendered_images, elev, azim, _, faces_idx = render.render_views(
                elev,
                azim,
                2,
                mesh,
                num_views=opt.view_num,
                show=False,
                center_azim=opt.frontview_center[0],
                center_elev=opt.frontview_center[1],
                std=opt.frontview_std,
                return_views=True,
                return_mask=True,
                return_face_idx=True,
                lighting=True,
                background=background,
                seed=2023,
            )
            save_pos(elev, azim, 2.0, save_path)
            print(f"elev:{elev} azim:{azim}")
            # print(f"Rendering the views...done")
            import torchvision
            from PIL import Image
            import numpy as np
            for i in range(len(rendered_images)):
                image = rendered_images[i]
                image_pil = torchvision.transforms.ToPILImage()(image)
                image_pil.save(f'{opt.output_dir}/step_1/output_image_{i}.png')
                torch.save(faces_idx[i], f'{opt.output_dir}/step_1/output_image_{i}.pt')

    if opt.step == "step_3":
        print("The code is currently running: Segment Mesh")
        import cv2
        import pickle
        import numpy as np
        from PIL import Image
        import pandas as pd
        save_path = opt.step2_data.replace('step_2', 'step_3')
        ensure_directory(save_path)
        data_path = os.path.join(opt.output_dir, 'step_1')

        # Load render image's face id
        faces_idx = []
        names = [file for file in os.listdir(data_path) if file.endswith('.png')]
        for name in names:
            image_ = os.path.join(data_path, name)
            face_ = torch.load(image_.replace('png', 'pt'))
            faces_idx.append(face_)
        print(f"load step 1 face id : {len(faces_idx)}")
        rendered_images = len(faces_idx)

        # Create the segmenter
        segmenter = Reason3D(opt)
        segmenter.set_rendered_views(rendered_images, faces_idx)
        segmenter.set_mesh(mesh)
        segmenter.set_prompts(opt.input_prompt)

        # Load render image mask bbox and scores
        res = []
        names = [file for file in os.listdir(opt.step2_data) if file.endswith('mask_0.jpg')]
        for name in names:
            image_ = os.path.join(opt.step2_data, name)
            if not os.path.exists(image_):
                continue
            img = Image.open(image_)
            img = np.array(img)

            threshold_value = 80
            _, img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
            with open(image_.replace('mask_0.jpg', 'boxlist_0.pkl'), 'rb') as f:
                bbox = pickle.load(f)
            if '8' in name or '9' in name:
                bbox['scores'] = torch.tensor([0.99] * len(bbox['scores']))
            part = (bbox['scores'], bbox['boxes'], img)
            res.append(part)
        print(f"load step 2 input image : {len(res)}")

        # Get segment mesh pre
        predictions, _ = segmenter(res)
        predictions = torch.tensor(predictions)
        print("sucess deal step2 data!")

        # Save the predictions
        np.save(os.path.join(save_path, "raw_face_preds.npy"), predictions.cpu().numpy())
        print("sucess save npy!")

        # threshold
        data_flat = predictions.flatten()
        data_series = pd.Series(data_flat)
        description = data_series.describe()
        socre = description['mean']

        faces_not_assigned = torch.where(torch.sum(predictions, axis=-1) <= socre)[0]
        predictions_cls = predictions.argmax(axis=-1)
        predictions_cls[faces_not_assigned] = len([opt.input_prompt])

        cols = colors_lst
        cols += [[0.5, 0.5, 0.0]]
        # cols[:-1] lable color cols[-1]: [0.5, 0.5, 0.0] : unknow lable color

        # Color the divided faces
        segmenter.color_mesh_and_save(
            predictions_cls, cols, os.path.join(save_path, "seg_model.obj")
        )
        print("sucess save obj!")

        # The predictions now are saved as strings
        with open(os.path.join(save_path, "face_preds.json"), "w") as fout:
            faces_cls = []
            for el in list(predictions_cls.cpu().numpy().astype(int)):
                faces_cls.append(cls_id_to_name[int(el)])
            json.dump(faces_cls, fout)

        with open(osp.join(save_path, "vert_preds.json"), "w") as fout:
            vertices_cls = [""] * len(mesh.vertices)
            for i_f, f_vs in enumerate(mesh.faces):
                for v in f_vs:
                    vertices_cls[v] = cls_id_to_name[int(predictions_cls[i_f])]

            json.dump(vertices_cls, fout)

