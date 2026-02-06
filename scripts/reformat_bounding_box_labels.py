#!/usr/bin/env python3
import os
import glob
import cv2
from tqdm import tqdm

# absolute path to dataset images
BASE_PATH = "/home/daksh/drone-thermal-detection/datasets/train/images"

# people = 1, bikes = 2, vehicles = 0
CLASS_MAPPING = {
    "pedestrian": 1,
    "person": 1,
    "human": 1,
    
    "elebike": 2,
    "tricycle": 2,
    "bike": 2,

    "c-vehicle": 0,
    "train": 0,
    "truck": 0,
    "bus": 0,
    "car": 0,
    "vehicle": 0
}

def get_class_id(folder_name):
    name_lower = folder_name.lower() # lowercase string
    for key, class_id in CLASS_MAPPING.items(): # searchs class keys for hit
        if key in name_lower:
            return class_id
    return -1 # failed

def process_sequence(seq_path):
    folder_name = os.path.basename(seq_path)
    class_id = get_class_id(folder_name)

    if class_id == -1:
        print(f"Skipping {folder_name}: Unknown class.")
        return

    # File paths
    rgb_txt_path = os.path.join(seq_path, "rgb.txt")
    labels_dir = os.path.join(seq_path, "labels")
    fused_img_dir = os.path.join(seq_path, "fused_images")

    if not os.path.exists(rgb_txt_path):
        return # if no rgb.txt, skip

    # Create output folder
    os.makedirs(labels_dir, exist_ok=True)

    # Read all lines from rgb.txt
    with open(rgb_txt_path, 'r') as f:
        # Filter out empty lines just in case
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # rgb.txt has bounding boxes for every 10th frame, I took every 30th frame
    # so use every 3rd line
    for line_idx in range(0, len(lines), 3):
        
        frame_num = line_idx * 10
        file_id = f"{frame_num:06d}" #  e.g. "000030"
        
        image_filename = f"{file_id}.jpg"
        image_path = os.path.join(fused_img_dir, image_filename)

        if not os.path.exists(image_path):
            continue # image doesn't exist, skip

        img = cv2.imread(image_path)
        if img is None:
            continue # get dimensions, skip if failed
        
        img_h, img_w = img.shape[:2]

        # get bounding box coordinates [x_min, y_min, width, height]
        try:
            raw_coords = lines[line_idx].split()
            x_min = float(raw_coords[0])
            y_min = float(raw_coords[1])
            w_pixel = float(raw_coords[2])
            h_pixel = float(raw_coords[3])
        except (ValueError, IndexError):
            print(f"Error parsing line {line_idx} in {folder_name}")
            continue

        # 5. Convert to YOLO Format 
        # x_center = (x_min + width/2) / total_width
        x_center = (x_min + (w_pixel / 2.0)) / img_w
        y_center = (y_min + (h_pixel / 2.0)) / img_h
        w_norm = w_pixel / img_w
        h_norm = h_pixel / img_h

        # normalize bounding box to [0,1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        # write to new label file and rename fused image to match
        final_name = f"{folder_name}_{file_id}"
        label_filename = f"{final_name}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as out:
            # YOLO Format: <class_id> <x_center> <y_center> <width> <height>
            out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        new_fused_image_name = f"{final_name}.jpg" # match label to fused_image
        new_fused_image_path = os.path.join(fused_img_dir, new_fused_image_name) # got it loaded, might as well rename while I'm at it
        if  image_path != new_fused_image_path:
            os.rename(image_path, new_fused_image_path) # rename fused image to match label
    

def main():
    # sorted list of drone run folders, though atp idk if that matters
    sequences = sorted([d for d in glob.glob(os.path.join(BASE_PATH, "*")) if os.path.isdir(d)])
    
    print(f"Found {len(sequences)} sequences. Starting label generation...")
    
    for seq in tqdm(sequences):
        process_sequence(seq)
        
    print("Done! Labels created in /labels/ subdirectories.")

if __name__ == "__main__":
    main()
