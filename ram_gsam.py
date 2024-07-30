import argparse
import os

# PATH = '/home/dyyao2/scratch'
# os.environ['HF_HOME'] = PATH

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm

import sys
sys.path.append("Grounded-Segment-Anything")

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import Model

# segment anything
from segment_anything import (
    build_sam,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

import supervision as sv
from supervision.draw.color import Color, ColorPalette
import dataclasses


remove_classes = set([])

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    caption = caption.split(",")
    classes = []
    for c in caption:
        if c.strip() in remove_classes:
            continue
        no_add = False
        for s in c.split():
            if s in remove_classes:
                no_add = True
                break
        if not no_add:
            classes.append(c.strip()) # lower case or else exception will occur

    with torch.no_grad():
        detections = model.predict_with_classes(
                image=image, # This function expects a BGR image...
                classes=classes,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

    return detections, classes

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def vis_result_fast(
    image: np.ndarray, 
    detections: sv.Detections, 
    classes, 
    color = ColorPalette.default(), 
    instance_random_color= False,
    draw_bbox = True,
) -> np.ndarray:
    '''
    Annotate the image with the detection results. 
    This is fast but of the same resolution of the input image, thus can be blurry. 
    '''
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        color = color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(
        color = color
    )

    labels = [
        f"{classes[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    
    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
        
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    
    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image, labels

def id_to_colors(id): # id to color
        rgb = np.zeros((3, ), dtype=np.uint8)
        for i in range(3):
            rgb[i] = id % 256
            id = id // 256
        return rgb

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--work_dir", type=str, default="/projects/perception/datasets/scannet200/ovir_preprocessed_data_val/scans")
    parser.add_argument("--out_dir",  type=str, default="/projects/perception/datasets/scannet200/ovir_preprocessed_data_val/scans")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    split = args.split
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    work_dir = args.work_dir
    out_dir = args.out_dir

    # FILTER = set(["person", "man", "boy", "couple", "skateboarder"])

    print(f"Working on {work_dir}")

    # ------------------ load model ------------------ #

    # model = load_model(config_file, grounded_checkpoint, device=device)
    grounding_dino_model = Model(
        model_config_path=config_file, 
        model_checkpoint_path=grounded_checkpoint, 
        device=device
    )

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    ram_model = ram(pretrained=ram_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    ram_model.eval()

    ram_model = ram_model.to(device)

    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    idx_to_id = [i for i in range(256*256*256)]
    np.random.shuffle(idx_to_id) # mapping to randomize idx to id to get random color

    # ------------------ ------------------------------------ #

    # make dir
    assert(os.path.exists(work_dir))

    scans = sorted(os.listdir(work_dir))
    # scans = [""]

    all_tags = []

    for i,scan in enumerate(scans):

        work_dir_scan = os.path.join(work_dir, scan)
        out_dir_scan = os.path.join(out_dir, scan)
        output_scan = os.path.join(out_dir_scan, 'ram_gsam_window')
        # output_scan = os.path.join(work_dir_scan, 'ram_gsam_window_all')
        images_path = os.path.join(work_dir_scan, "color")
        
        os.makedirs(os.path.join(output_scan, "mask"), exist_ok=True)
        os.makedirs(os.path.join(output_scan, "vis"), exist_ok=True)

        # if os.path.exists(os.path.join(output_scan, "done.txt")):
        #     continue

        print(f"Working on {scan}", flush=True)

        active_tags = {} # counter to keep tags alive in window

        for image_name in sorted(os.listdir(images_path)):

            image_path = os.path.join(images_path, image_name)

            # load image
            image_pil = Image.open(image_path).convert("RGB")
    
            raw_image = image_pil.resize(
                            (384, 384))
            raw_image  = transform(raw_image).unsqueeze(0).to(device)

            res = inference_ram(raw_image , ram_model)

            # Currently ", " is better for detecting single tags
            # while ". " is a little worse in some case
            tags=res[0].split(' |')

            for tag in tags:
                active_tags[tag] = 50

            removes = []
            for tag in active_tags:
                if tag not in tags:
                    active_tags[tag] = active_tags[tag] - 1
                    if active_tags[tag] == 0:
                        removes.append(tag)
            
            for tag in removes:
                del active_tags[tag]
     
            if len(active_tags) > 100:
                print("MORE THAN 100 ACTIVE TAGS")
            tags = ",".join(list(active_tags.keys()))

            image_path = os.path.join(images_path, image_name)

            # load image
            image_pil = Image.open(image_path).convert("RGB")

            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            image_rgb = np.array(image_pil)

            # run grounding dino model
            detections, classes = get_grounding_output(
                grounding_dino_model, image, tags, box_threshold, text_threshold, device=device
            )

            valid_idx = np.logical_and(detections.class_id != -1, detections.class_id != None)
            too_big = detections.area > 0.8 * image_rgb.shape[0] * image_rgb.shape[1]
            valid_idx = np.logical_and(valid_idx, np.logical_not(too_big))
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]

            if len(detections.class_id) > 0:
                nms_idx = torchvision.ops.nms(
                            torch.from_numpy(detections.xyxy), 
                            torch.from_numpy(detections.confidence), 
                            iou_threshold
                        ).numpy().tolist()

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]

                ### Segment Anything ###
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )

                annotated_image, labels = vis_result_fast(
                    image, detections, classes, instance_random_color=True)

                masks = detections.mask
                labels = detections.class_id

                assert(np.sum(labels == -1) == 0) # check if any label == -1? concept graph has a bug with this

                color_mask = np.zeros(image_rgb.shape, dtype=np.uint8)

                obj_info_json = []

                #sort masks according to size
                mask_size = [np.sum(mask) for mask in masks]
                sorted_mask_idx = np.argsort(mask_size)[::-1]

                for idx in sorted_mask_idx: # render from largest to smallest
                    
                    mask = masks[idx]
                    color_mask[mask] = id_to_colors(idx_to_id[idx])

                    obj_info_json.append({
                        "id": idx_to_id[idx],
                        "label": classes[labels[idx]],
                        "score": float(detections.confidence[idx]),
                    })

                color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(output_scan, "vis", image_name), annotated_image) # VISUALIZATION
                image_name = image_name.replace(".jpg", ".png")
                cv2.imwrite(os.path.join(output_scan, "mask", image_name), color_mask)
                with open(os.path.join(output_scan, "mask", image_name.replace(".png", ".json")), "w") as f:
                    json.dump(obj_info_json, f)
            
            else:
                
                cv2.imwrite(os.path.join(output_scan, "vis", image_name), image) # VISUALIZATION
                image_name = image_name.replace(".jpg", ".png")
                cv2.imwrite(os.path.join(output_scan, "mask", image_name), np.zeros(image.shape, dtype=np.uint8))
                with open(os.path.join(output_scan, "mask", image_name.replace(".png", ".json")), "w") as f:
                    json.dump([], f)

            print(f"Done with {image_name}", flush=True)

        with open(os.path.join(output_scan, "done.txt"), "w") as f:
            pass
