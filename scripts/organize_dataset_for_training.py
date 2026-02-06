#!/usr/bin/env python3

# Organizes and flattens dataset into train/val folders
# Randomly picks 20% of data for validation, rest for training
import os
import glob
import shutil
import random
from tqdm import tqdm


SOURCE_ROOT = "/home/daksh/drone-thermal-detection/datasets/train/images"
DEST_ROOT = "/home/daksh/drone-thermal-detection/datasets/final_yolo" # new, otherwise would be so messy

VAL_RATIO = 0.2 # 20% for validation
SEED = 42  # makes same random split every time for training consistency

def main():
    # sets up file structure for YOLO training
    train_img_dir = os.path.join(DEST_ROOT, "train", "images")
    train_lbl_dir = os.path.join(DEST_ROOT, "train", "labels")
    val_img_dir = os.path.join(DEST_ROOT, "val", "images")
    val_lbl_dir = os.path.join(DEST_ROOT, "val", "labels")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("Scanning for label files...")
    # finds all .txt files in OG dataset
    all_label_paths = glob.glob(os.path.join(SOURCE_ROOT, "*", "labels", "*.txt"))
    
    data_pairs = [] # O(n!) here i fucking come lolz
    
    print(f"Found {len(all_label_paths)} label files. verifying images...")
    
    for label_path in tqdm(all_label_paths):
        # get image path from file path
        
        # go up two levels to sequence folder (from /labels/ to /bus_007/)
        seq_folder = os.path.dirname(os.path.dirname(label_path))
        
        filename_base = os.path.splitext(os.path.basename(label_path))[0]
        image_name = f"{filename_base}.jpg"
        
        # find corresponding fused image
        image_path = os.path.join(seq_folder, "fused_images", image_name)
        
        if os.path.exists(image_path):
            data_pairs.append((image_path, label_path))
        else:
            print(f"Warning: Image missing for {label_path}")

    random.seed(SEED) # shuffles so random frames
    random.shuffle(data_pairs)

    # split into train and val
    split_index = int(len(data_pairs) * (1 - VAL_RATIO))
    train_pairs = data_pairs[:split_index]
    val_pairs = data_pairs[split_index:]

    print(f"Total Data: {len(data_pairs)}")
    print(f"Training: {len(train_pairs)} images")
    print(f"Validation: {len(val_pairs)} images")

    # copy and paste image/label pairs into new train/val folders
    def copy_set(pairs, img_dest, lbl_dest, desc):
        for img_src, lbl_src in tqdm(pairs, desc=desc):
            shutil.copy(img_src, os.path.join(img_dest, os.path.basename(img_src)))
            shutil.copy(lbl_src, os.path.join(lbl_dest, os.path.basename(lbl_src)))

    copy_set(train_pairs, train_img_dir, train_lbl_dir, "Copying Train")
    copy_set(val_pairs, val_img_dir, val_lbl_dir, "Copying Val")
    
    # 6. Generate data.yaml
    yaml_content = f"""path: {DEST_ROOT}
train: train/images
val: val/images

nc: 3
names:
  0: vehicle
  1: person
  2: cycle
"""
    yaml_path = os.path.join(DEST_ROOT, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"\nSuccess! Dataset ready at {DEST_ROOT}")
    print(f"Your data.yaml is located at {yaml_path}")

if __name__ == "__main__":
    main()
