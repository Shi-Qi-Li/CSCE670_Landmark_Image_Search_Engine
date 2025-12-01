import os
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from landmark_head import LandmarkHead


def parse_args():
    parser  = argparse.ArgumentParser(description="Landmark Dataset Preprocessing")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_dim", type=int, default=384)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--multihead_num", type=int, default=4)
    parser.add_argument("--layers_num", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main(args):
    dataset_dir = args.dataset_dir
    device = args.device
    
    # model config
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    multihead_num = args.multihead_num
    layers_num = args.layers_num
    
    landmark_head = LandmarkHead(input_dim=input_dim, hidden_dim=hidden_dim, multihead_num=multihead_num, layers_num=layers_num, device=device)
    landmark_head.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state_dict"])
    landmark_head.eval()


    dataset_preprocessed_dir = os.path.join(dataset_dir, "preprocessed")
    for file_name in tqdm(os.listdir(dataset_preprocessed_dir)):
        embeddings = torch.load(os.path.join(dataset_preprocessed_dir, file_name), map_location="cpu")

        cls_tokens = embeddings["cls_token"].to(device)  # [1, input_dim]
        patch_features = embeddings["patch_features"].to(device)  # [1, num_patches, input_dim]
        
        with torch.no_grad():
            retrieval_feature = landmark_head(cls_tokens, patch_features)  # [1, hidden_dim]
        retrieval_feature = F.normalize(retrieval_feature, p=2, dim=1)

        embeddings.update({"retrieval_feature": retrieval_feature.cpu()})    
            
        output_path = os.path.join(dataset_preprocessed_dir, file_name)
        torch.save(embeddings, output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)