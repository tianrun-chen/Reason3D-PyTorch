## Reasoning3D - Grounding and Reasoning in 3D: Fine-Grained Zero-Shot Open-Vocabulary 3D Reasoning Part Segmentation via Large Vision-Language Models 



## Installation

1. Create Environment
```conda create -n 3Dreason python=3.8
conda activate 3Dreaso
```

2. Install necessary packages
```pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html
pip install flash-attn --no-build-isolation
```

```pip install -r requirements.txt
```

3. Install rely package
```pip install -e .
```

##Datasets

For FAUST, click this line to download the FAUST dataset, and put it in input/
    https://drive.google.com/drive/folders/1T5reNd6GqRfQRyhw8lmhQwCVWLcCOZVN
For our reasoning 3D segmentation data, you can play with your own interested models. We collect models from Sketchfab. 


##Pre-trained weights

Download thie weights in pre_model/

https://huggingface.co/xinlai/LISA-13B-llama2-v0-explanatory


##Running

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

##Evaluation

Given an output dir containing the coarse predictions for the len(mesh_name.txt) scans.
run coarse as following:
```python evaluate.py -output_dir outputs/ -mesh_name input/FAUST/mesh_name.txt
```

run fine_grained as following:
```python evaluate.py -fine_grained -output_dir outputs/ -mesh_name input/FAUST/mesh_name.txt
```


