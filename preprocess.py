import argparse
import os
from PIL import Image

import pandas as pd
import torch
from tqdm import tqdm

from image_encoder import ImageEncoder

def parse_args():
    parser  = argparse.ArgumentParser(description="Landmark Dataset Preprocessing")
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dino_lib', type=str, default='facebook/dinov3-vit7b16-pretrain-lvd1689m')
    return parser.parse_args()

def main(args):
    dataset_dir = args.dataset_dir
    device = args.device
    
    # load image encoder
    image_encoder = ImageEncoder(lib=args.dino_lib, device=device)
    
    # load dataset
    image_dataset_dir = os.path.join(dataset_dir, 'train/0')
    meta_data_dir = os.path.join(dataset_dir, 'train.csv') # 3 columns: image_id(file name), url, landmark_id
    output_dir = os.path.join(dataset_dir, 'preprocessed')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_dir_list_level1 = os.listdir(image_dataset_dir)
    dataset_dir_list = []
    for dir in dataset_dir_list_level1:
        dataset_dir_list_level2 = os.listdir(os.path.join(image_dataset_dir, dir))
        for dir_level2 in dataset_dir_list_level2:
            data_dir_list_level3 = os.listdir(os.path.join(image_dataset_dir, dir, dir_level2))
            for data_dir_level3 in data_dir_list_level3:
                data_dir = os.path.join(image_dataset_dir, dir, dir_level2, data_dir_level3)
                dataset_dir_list.append(data_dir)
    
    for data_dir in tqdm(dataset_dir_list):
        image_id = os.path.splitext(os.path.basename(data_dir))[0]
        image = Image.open(data_dir).convert('RGB')
        embeddings = image_encoder.encode([image])
        output_path = os.path.join(output_dir, f"{image_id}.pt")
        torch.save(embeddings, output_path)
                
if __name__ == '__main__':
    args = parse_args()
    main(args)