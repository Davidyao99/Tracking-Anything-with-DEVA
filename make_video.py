import cv2
import os
import numpy as np
import json


def vis_num(frame, mask):

    obj_idxs = np.sort(np.unique(mask))
    obj_idxs = obj_idxs[obj_idxs!=0]

    color = (255,255,255)

    for i, idx in enumerate(obj_idxs):
        centroid = np.mean(np.argwhere(mask==idx),axis=0)
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])

        cv2.putText(frame, str(ids_to_idx_dict[idx]), (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, 2)

    return frame

def _id_to_rgb(id_mask):
    ids = np.unique(id_mask)
    res_frame = np.zeros((id_mask.shape[0], id_mask.shape[1], 3), dtype=np.uint8)
    for id in ids:
        id_copy = id
        rgb = np.zeros((3, ), dtype=np.uint8)
        for i in range(3):
            rgb[i] = id_copy % 256
            id_copy = id_copy // 256
        res_frame[id_mask == id] = rgb

    return res_frame

# Replace 'input_directory' with the path to the directory containing frames
work_dir = '/projects/perception/personals/david/OVIR-3D_V1/ScanNet_val_split/'
model = 0

if model == 0:
    deva_path = "results/deva_gsam"
elif model == 1:
    deva_path = "results/deva_mask2former"
elif model == 2:
    deva_path = "results/deva_detic"

videos = sorted(os.listdir(work_dir))

for video in videos[:1]:

    input_directory = os.path.join(work_dir, video, deva_path, "Annotations")
    input_frame_dir = os.path.join(work_dir, video, "color")
    pred_json_path = os.path.join(work_dir, video, deva_path, "pred.json")

    id_dict = json.load(open(pred_json_path, 'r'))
    id_dict = id_dict['annotations']

    all_ids = []
    s = set([])
    for file in id_dict:
        for segments in file['segments_info']:
            if segments['id'] not in s:
                s.add(segments['id'])
                all_ids.append(segments['id'])
    print("Total of {} ids".format(len(all_ids)))
    ids_to_idx_dict = {id: idx+1 for idx, id in enumerate(all_ids)}


    # Replace 'output_video.mp4' with the desired output video file name
    output_video_path = os.path.join(work_dir, video, deva_path, "output_video.mp4")

    # Get the list of image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]
    frame_files = [f for f in os.listdir(input_frame_dir) if f.endswith('.jpg')]

    # Sort the image files to ensure the frames are in the correct order
    image_files.sort()
    frame_files.sort()

    # Open the first image to get its dimensions
    first_image_path = os.path.join(input_directory, image_files[0])
    first_image = np.load(first_image_path)
    height, width = first_image.shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID', 'MJPG', or other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    # Loop through the image files and add them to the video
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_directory, image_file)
        frame_path = os.path.join(input_frame_dir, frame_files[i])

        image_npy = np.load(image_path)
        mask = _id_to_rgb(image_npy)
        frame = cv2.imread(frame_path)

        frame = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)
        vis_num(frame, image_npy)

        cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, 2)

        out.write(frame)

        if i%10 == 0:
            print(f"Added frame {i}", flush=True)

    # Release the VideoWriter object
    out.release()

    print(f"Video saved as {output_video_path}")
