import os
from os import path
from argparse import ArgumentParser

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame_with_text as process_frame

from tqdm import tqdm
import json

import clip

if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)

    # for id2rgb
    np.random.seed(42)
    """
    Arguments loading
    """
    parser = ArgumentParser()

    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    deva_model, cfg, args = get_model_and_config(parser)
    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')

    clip_model, clip_preprocess = clip.load("ViT-L/14", device='cuda', download_root="/home/dyyao2/scratch/")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    with open(f"{args.output}/args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    """
    Temporal setting
    """
    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']

    # get data
    video_reader = SimpleVideoReader(cfg['img_path'])
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
    out_path = cfg['output']

    test_done_path = os.path.join(out_path, "done.txt")
    if os.path.exists(test_done_path):
        print(f"We are done with images at {cfg['img_path']}, skip it.")
        exit()

    # Start eval
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    cfg['enable_long_term_count_usage'] = (
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
             cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    print('Configuration:', cfg)

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            process_frame(deva, gd_model, sam_model, clip_model, clip_preprocess, im_path, result_saver, ti, image_np=frame)
            # process_frame(deva, gd_model, sam_model, im_path, result_saver, ti, image_np=frame)
        flush_buffer(deva, result_saver)
    result_saver.end()

    # save this as a video-level json
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json

    obj_summary = result_saver.get_all_obj_summary()

    with open(path.join(out_path, 'tracklets.json'), 'w') as f:
        json.dump(obj_summary, f, indent=4)  # prettier json

    Path(os.path.join(out_path, "done.txt")).touch()