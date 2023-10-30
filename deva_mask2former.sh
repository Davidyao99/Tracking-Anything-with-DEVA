#!/bin/bash

# Check if a directory path is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

# Store the parent directory and Python script in variables
parent_directory="$1"


# Check if the parent directory is a valid directory
if [ ! -d "$parent_directory" ]; then
    echo "$parent_directory is not a valid directory."
    exit 1
fi

# Use the find command to locate all directories in the parent directory
# and pass each directory as an argument to the Python script
find "$parent_directory" -mindepth 1 -maxdepth 1 -type d | sort | while read -r directory; do
    if [ $((counter % $2)) -eq  $3 ]; then
    # Call the Python script with the directory as an argument
    echo "working on $counter : $directory"
    fbname=$(basename "$directory")
    echo "saving to $fbname"
    python3 demo/demo_with_mask2former.py --chunk_size 4 --img_path "$directory/color" --amp --temporal_setting semionline --max_missed_detection_count 2 --size -1 --output "$directory/results/deva_mask2former_test" --config-file /projects/perception/personals/david/Tracking-Anything-with-DEVA/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_swin_large_3x.yaml --opts MODEL.WEIGHTS /projects/perception/personals/david/Tracking-Anything-with-DEVA/detectron2/projects/CropFormer/checkpoints/Mask2Former_swin_large_w7_3x_dd4543.pth
    fi
    ((counter++))
done