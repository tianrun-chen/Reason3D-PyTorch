## Reasoning3D - Grounding and Reasoning in 3D: Fine-Grained Zero-Shot Open-Vocabulary 3D Reasoning Part Segmentation via Large Vision-Language Models 

<a href="http://tianrun-chen.github.io/" target="_blank">Tianrun Chen</a>, Chun'an Yu, Jing Li, Lanyun Zhu, Deyi Ji, Yong Zhang, Ying Zang, Zejian Li, Lingyun Sun

<a href='https://www.kokoni3d.com/'> KOKONI, Moxin Technology (Huzhou) Co., LTD </a>, Zhejiang University, Singapore University of Technology and Design, Huzhou University, University of Science and Technology of China.

<img src='https://tianrun-chen.github.io/Reason3D/static/images/Fig1.jpg'>

## Installation
0. git clone this repository :D
   
1. Create Environment
```conda create -n 3Dreason python=3.8
conda activate 3Dreaso
```

2. Install necessary packages
```pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install -e .
```
3. Add your customized data in input/ folder
   
4. Download thie weights in pre_model/
https://huggingface.co/xinlai/LISA-13B-llama2-v0-explanatory

5. run main.py. You will see a user interface like this:
<img src='https://tianrun-chen.github.io/Reason3D/static/images/ui.jpg'>

## Datasets
For our reasoning 3D segmentation data, you can play with your own interested models. We collect models from Sketchfab. https://sketchfab.com/ 

For FAUST dataset we use as the benchmark (for open-vocabulary segmentation), click this line to download the FAUST dataset, and put it in input/
    https://drive.google.com/drive/folders/1T5reNd6GqRfQRyhw8lmhQwCVWLcCOZVN/




## Running

For one FAUST data on the coarse segmentation:(tr_scan_000 example):
    1. mesh render image
    bash FAUST/gen_rander_img.sh
    Note: You can change pose.txt to what you want or change -frontview_center
    2. image get mask
    bash FAUST/gen_mask.sh
    3. image and mask gen seg mesh
    bash FAUST/gen_Seg_mesh.sh

For other data segmentation:
    1. mesh render image
    bash scripts/gen_rander_img.sh
    2. image get mask
    bash scripts/gen_mask.sh
    3. image and mask gen seg mesh
    bash scripts/gen_Seg_mesh.sh

## Qualitative Evaluation
Given an output dir containing the coarse predictions for the len(mesh_name.txt) scans.
run coarse as following:
```shell
python scripts/evaluate_faust.py -output_dir outputs/coarse_output_dir
```
or for the fine_grained:

```shell
python scripts/evaluate_faust.py --fine_grained -output_dir outputs/fine_grained_output_dir
```

## Acknowledgements
The part of the code is derived from SATR: Zero-Shot Semantic Segmentation of 3D Shapes <a href='https://github.com/Samir55/SATR'><img src='https://img.shields.io/badge/Project-Page-Green'></a> by Ahmed Abdelreheem, Ivan Skorokhodov, Maks Ovsjanikov, and Peter Wonka
from KAUST and LIX, Ecole Polytechnique. Thanks to the authors for their awesome work!


