# Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything

from typing import Dict, List
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torchvision

from deva.inference.object_info import ObjectInfo
from PIL import Image

# def get_hidden_states_clip(clip_model, clip_preprocess, image, boxes):

#     crops = [clip_preprocess(Image.fromarray(image))]
#     h,w = image.shape[:2]
#     for box in boxes:
#         crop = image[max(0,int(box[1])):min(h,int(box[3])), max(0,int(box[0])): min(w,int(box[2]))]
#         crop_image = Image.fromarray(crop)
#         crop = clip_preprocess(crop_image)
#         crops.append(crop)

#     if len(crops) == 0:
#         return []

#     crops_torch = torch.from_numpy(np.stack(crops)).to('cuda')

#     with torch.no_grad():
#         logits = clip_model.encode_image(crops_torch)

#     global_feat = torch.nn.functional.normalize(logits[0][None,:], dim=-1)
#     crop_feats = torch.nn.functional.normalize(logits[1:], dim=-1)

#     sim = (global_feat @ crop_feats.T)

#     softmax_scores = torch.nn.functional.softmax(sim, dim=1).T
#     weighted_clip = softmax_scores * global_feat + (1 - softmax_scores) * crop_feats

#     torch.cuda.empty_cache()

#     return weighted_clip.cpu().numpy()

def make_segmentation_with_custom(cfg, image, seg_model):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    vis_output = None
    predictions = seg_model(image)
    # Convert image from OpenCV BGR format to Matplotlib RGB format.
    image = image[:, :, ::-1]
    assert("instances" in predictions)
    predictions = predictions["instances"].to(torch.device("cpu"))

    pred_masks = predictions.pred_masks
    pred_scores = predictions.scores
        
    # select by confidence threshold
    selected_indexes = (pred_scores >= cfg['custom_seg_threshold']).detach().numpy()
    selected_scores = pred_scores[selected_indexes].detach().numpy()
    selected_masks  = pred_masks[selected_indexes].detach().numpy()
    areas = selected_masks.sum(axis=(1,2))
    
    _, m_H, m_W = selected_masks.shape

    selected_indexes = areas > cfg['custom_mask_filter_threshold'] # filter out small masks

    selected_scores = selected_scores[selected_indexes]
    selected_masks  = selected_masks[selected_indexes]

    boxes = []

    for selected_mask in selected_masks:
        horizontal_indicies = np.where(np.any(selected_mask, axis=0))[0]
        vertical_indicies = np.where(np.any(selected_mask, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        boxes.append((x1, y1, x2, y2))

    # hidden_states = get_hidden_states_clip(clip_model, clip_preprocess, image, boxes)

    output_mask = torch.zeros((m_H, m_W), dtype=torch.int64, device='cuda')

    curr_id = 1
    segments_info = []

    for i in range(len(selected_masks)):
        segments_info.append(ObjectInfo(id=curr_id, category_id=0, score=selected_scores[i].item()))
        output_mask[selected_masks[i] > 0] = curr_id
        curr_id += 1

    return output_mask, segments_info


 
# def segment_with_text(config: Dict, gd_model: GroundingDINOModel, sam: SamPredictor, clip_model, 
#                     clip_preprocess,
#                       image: np.ndarray, prompts: List[str],
#                       min_side: int) -> (torch.Tensor, List[ObjectInfo]):
#     """
#     config: the global configuration dictionary
#     image: the image to segment; should be a numpy array; H*W*3; unnormalized (0~255)
#     prompts: list of class names

#     Returns: a torch index mask of the same size as image; H*W
#              a list of segment info, see object_utils.py for definition
#     """

#     BOX_THRESHOLD = TEXT_THRESHOLD = config['DINO_THRESHOLD']
#     NMS_THRESHOLD = config['DINO_NMS_THRESHOLD']

#     sam.set_image(image, image_format='RGB')

#     # detect objects
#     # GroundingDINO uses BGR
#     detections, hidden_states_gdino = gd_model.predict_with_classes(image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
#                                                classes=prompts,
#                                                box_threshold=BOX_THRESHOLD,
#                                                text_threshold=TEXT_THRESHOLD)

#     nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy),
#                                   torch.from_numpy(detections.confidence),
#                                   NMS_THRESHOLD).numpy().tolist()

#     detections.xyxy = detections.xyxy[nms_idx]
#     detections.confidence = detections.confidence[nms_idx]
#     detections.class_id = detections.class_id[nms_idx]

#     hidden_states_gdino = hidden_states_gdino[nms_idx]
#     hidden_states_clip = get_hidden_states_clip(clip_model, clip_preprocess, image, detections)

#     result_masks = []
#     for box in detections.xyxy:
#         masks, scores, _ = sam.predict(box=box, multimask_output=True)
#         index = np.argmax(scores)
#         result_masks.append(masks[index])

#     detections.mask = np.array(result_masks)

#     h, w = image.shape[:2]
#     if min_side > 0:
#         scale = min_side / min(h, w)
#         new_h, new_w = int(h * scale), int(w * scale)
#     else:
#         new_h, new_w = h, w

#     output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=gd_model.device)
#     curr_id = 1
#     segments_info = []

#     # sort by descending area to preserve the smallest object
#     for i in np.flip(np.argsort(detections.area)):
#         mask = detections.mask[i]
#         confidence = detections.confidence[i]
#         class_id = detections.class_id[i]
#         hidden_state_gdino = hidden_states_gdino[i]
#         hidden_state_clip = hidden_states_clip[i]

#         mask = torch.from_numpy(mask.astype(np.float32))
#         mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='bilinear')[0, 0]
#         mask = (mask > 0.5).float()

#         if mask.sum() > 0:
#             output_mask[mask > 0] = curr_id
#             segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence, 
#             hidden_state_seg=hidden_state_gdino, hidden_state_clip=hidden_state_clip))
#             curr_id += 1

#     return output_mask, segments_info