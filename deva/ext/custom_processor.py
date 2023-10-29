from os import path
from typing import Dict, List

import cv2
import torch
import numpy as np

from deva.inference.object_info import ObjectInfo
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.ext.custom_seg import make_segmentation_with_custom

def idx_to_color(idx):

    if idx == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    np.random.seed(idx) # randomly assign to some color, for visualizing gt
    id = np.random.randint(0, 256**3)


    rgb = np.zeros((3, ), dtype=np.uint8)
    for i in range(3):
        rgb[i] = id % 256
        id = id // 256
    return rgb 

def visualize_mask(mask):

    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for idx in np.unique(mask):
        mask_vis[mask==idx] = idx_to_color(idx)

    return mask_vis

@torch.inference_mode()
def process_frame_custom(deva, seg_model, clip_model, clip_preprocess, frame_path, result_saver, ti, image_np):

    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            mask, segments_info = make_segmentation_with_custom(cfg, image_np, seg_model, clip_model, 
                                                            clip_preprocess)
            
            # custom_mask = mask.clone().detach().cpu().numpy()
            # vis_mask = visualize_mask(custom_mask)
            # cv2.imwrite(f"/projects/perception/personals/david/OVIR-3D_V1/ScanNet/scene0011_01/deva_custom_detic/entityseg/entity_seg_{ti}.png", vis_mask)

            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image, mask, new_segments_info)
                deva.next_voting_frame += cfg['detection_every']

                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np,
                                       prompts=["nothing"])

                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np,
                                           prompts=["nothing"])

                deva.clear_buffer()
        else:
            # standard propagation
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np,
                                   prompts=["nothing"])

    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            mask, segments_info = make_segmentation_with_custom(cfg, image_np, seg_model, clip_model, 
                                                            clip_preprocess)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np,
                               prompts=["nothing"])
