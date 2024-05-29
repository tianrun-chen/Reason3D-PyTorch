import argparse
import os
import sys
import pickle
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor

import time
from model.LISA import LISA
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.conversation import get_default_conv_template


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--image_path", default="output/step_1")
    parser.add_argument("--prompt", default="Can you segment the supporting part in this image? Please output segmentation mask and explain why.")
    parser.add_argument("--version", default="pre_model/LISA-13B-llama2-v0-explanatory")
    parser.add_argument("--vis_save_path", default="output/step_2", type=str)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image-size", default=1024, type=int, help="image size")
    parser.add_argument("--model-max-length", default=512, type=int)
    parser.add_argument("--lora-r", default=-1, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def get_Boxlist(boxes, labels, scores, image_size):
    """Make Boxlist"""
    boxlist = {
        "boxes": boxes,  
        "labels": labels,  
        "scores": scores,  
        "image_size": image_size  
    }
    return boxlist

def convert_bbox_to_points(bounding_boxes):
    """Convert the bounding box information to the coordinates of the upper left and lower right corners"""
    points = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        bbox = (x1,y1,x2,y2)
        points.append(bbox)
    return points

def calculate_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    area = width * height
    return area

def filter_bboxes(bboxes):
    areas = [(calculate_bbox_area(bbox), bbox) for bbox in bboxes]
    areas.sort(reverse=True, key=lambda x: x[0])

    max_area = areas[0][0]
    new_bboxes = []

    for area, bbox in areas:
        if (max_area - area) <= 10000:      # Area check less than 10000
            new_bboxes.append(bbox)

    return new_bboxes

def get_bbox(mask, sorce):
    """Get bbox through mask"""
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes = convert_bbox_to_points(bounding_boxes)
    new_bboxes = filter_bboxes(bounding_boxes)

    return new_bboxes

def compute_colors_for_labels(labels):
    """Simple function that adds fixed colors depending on the class"""
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = (30 * (labels[:, None] - 1) + 1) * palette
    colors = (colors % 255).numpy().astype("uint8")
    try:
        colors = (colors * 0 + 255).astype("uint8")
    except:
        pass
    return colors

def overlay_boxes(image, bounding_boxes, prompt):
    labels = prompt
    boxes = bounding_boxes

    colors = compute_colors_for_labels(labels).tolist()
    new_image = image.copy()
    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        new_image = cv2.rectangle(
            new_image, tuple(top_left), tuple(bottom_right), tuple(colors[0]), 2)

    return new_image


def main(args):
    args = parse_args(args)
    print(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    print("load tokenizer success!")
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    ret_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids
    args.seg_token_idx = ret_token_idx[0]

    model = LISA(
        args.local_rank,
        args.seg_token_idx,
        tokenizer,
        args.version,
        args.lora_r,
        args.precision,
        vision_tower=args.vision_tower,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    if os.path.exists(args.version):
        model_dir = args.version
    else: # hack for cached pre-trained weights
        user_name, model_name = args.version.split("/")
        cache_dir = "{}/.cache/huggingface/hub/models--{}--{}".format(os.environ['HOME'], user_name, model_name)
        if os.path.exists(cache_dir):
            model1_dir = glob.glob("{}/snapshots/*/pytorch_model-visual_model.bin".format(cache_dir))
            model2_dir = glob.glob("{}/snapshots/*/pytorch_model-text_hidden_fcs.bin".format(cache_dir))
            if len(model1_dir) == 0 or len(model2_dir) == 0:
                raise ValueError("Pre-trained weights for visual_model or text_hidden_fcs do not exist in {}.".format(
                    cache_dir
                ))
            model1_dir = ["/".join(x.split("/")[:-1]) for x in model1_dir]
            model2_dir = ["/".join(x.split("/")[:-1]) for x in model2_dir]
            model_dir = list(set(model1_dir).intersection(set(model2_dir)))
            if len(model_dir) == 0:
                raise ValueError("Pre-trained weights for visual_model or text_hidden_fcs do not exist in {}.".format(
                    cache_dir
                ))
            model_dir = model_dir[0]
        else:
            raise ValueError("The path {} does not exists.".format(cache_dir))

    weight = {}
    visual_model_weight = torch.load(
        os.path.join(model_dir, "pytorch_model-visual_model.bin")
    )
    text_hidden_fcs_weight = torch.load(
        os.path.join(model_dir, "pytorch_model-text_hidden_fcs.bin")
    )
    weight.update(visual_model_weight)
    weight.update(text_hidden_fcs_weight)
    missing_keys, unexpected_keys = model.load_state_dict(weight, strict=False)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
    else:
        model = model.float().cuda()
    print("load model success!")

    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    image_token_len = 256

    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    # while True:
    for root, dirs, files in os.walk(args.image_path):
        for file in files:
            if file.endswith('.png'):
                image = os.path.join(root, file)

                conv = get_default_conv_template("vicuna").copy()
                conv.messages = []
                # prompt = input("Please input your prompt: ")
                prompt = args.prompt
                prompt = DEFAULT_IMAGE_TOKEN + " " + prompt
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], "")
                prompt = conv.get_prompt()

                # image_path = input("Please input the image path: ")
                image_path = image
                if not os.path.exists(image_path):
                    print("File not found in {}".format(image_path))
                    continue

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_size_list = [image.shape[:2]]
                if args.precision == "bf16":
                    images_clip = (
                        clip_image_processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                        .unsqueeze(0)
                        .cuda()
                        .bfloat16()
                    )
                elif args.precision == "fp16":
                    images_clip = (
                        clip_image_processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                        .unsqueeze(0)
                        .cuda()
                        .half()
                    )
                else:
                    images_clip = (
                        clip_image_processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                        .unsqueeze(0)
                        .cuda()
                        .float()
                    )
                images = transform.apply_image(image)
                resize_list = [images.shape[:2]]
                if args.precision == "bf16":
                    images = (
                        preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
                        .unsqueeze(0)
                        .cuda()
                        .bfloat16()
                    )
                elif args.precision == "fp16":
                    images = (
                        preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
                        .unsqueeze(0)
                        .cuda()
                        .half()
                    )
                else:
                    images = (
                        preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
                        .unsqueeze(0)
                        .cuda()
                        .float()
                    )

                input_ids = tokenizer(prompt).input_ids
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
                output_ids, pred_masks, sorce = model.evaluate(
                    images_clip,
                    images,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                )
                text_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
                text_output = (
                    text_output.replace(DEFAULT_IMAGE_PATCH_TOKEN, "")
                    .replace("\n", "")
                    .replace("  ", "")
                )

                print("text_output: ", text_output)
                for i, pred_mask in enumerate(pred_masks):
                    if pred_mask.shape[0] == 0:
                        continue

                    pred_mask = pred_mask.detach().cpu().numpy()[0]
                    pred_mask = pred_mask > 0
                    """get bbox and save"""
                    bbox = torch.tensor(get_bbox(pred_mask, sorce))
                    labels = torch.ones(bbox.shape[0], dtype=torch.int)
                    scores = torch.tensor([sorce[0][i].item()] * bbox.shape[0], dtype=torch.float)
                    image_size = (1024, 1024)
                    # clor = compute_colors_for_labels(labels).tolist()
                    bbox_img = overlay_boxes(image, bbox, labels)
                    Boxlist_ = get_Boxlist(bbox, labels, scores, image_size)
                    save_path = "{}/{}_boxlist_{}.pkl".format(
                        args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                    )
                    with open(save_path, 'wb') as f:
                        pickle.dump(Boxlist_, f)
                    save_path = "{}/{}_bbox_{}.png".format(
                        args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                    )
                    cv2.imwrite(save_path, bbox_img)
                    print("{} has been saved.".format(save_path))

                    save_path = "{}/{}_mask_{}.jpg".format(
                        args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                    )
                    cv2.imwrite(save_path, pred_mask * 100)
                    print("{} has been saved.".format(save_path))

                    save_path = "{}/{}_masked_img_{}.jpg".format(
                        args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                    )
                    save_img = image.copy()
                    save_img[pred_mask] = (
                        image * 0.5
                        + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                    )[pred_mask]
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, save_img)
                    print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
