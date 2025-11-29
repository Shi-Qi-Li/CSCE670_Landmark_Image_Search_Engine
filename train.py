import os
from PIL import Image
import time
import random
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from landmark_head import LandmarkHead
from tqdm import tqdm
import pandas as pd

def arg_parse():
    parser = argparse.ArgumentParser(description="Landmark Head Training")
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--multihead_num', type=int, default=4)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_epochs', type=int, default=10)
    
    return parser.parse_args()
class Landmark_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, device='cpu'):
        self.device = device
        self.dataset_preprocessed_dir = os.path.join(dataset_dir, 'preprocessed')
        meta_data_dir = os.path.join(dataset_dir, 'train.csv') # 3 columns: image_id(file name), url, landmark_id
        
        # Load metadata CSV to create image_id -> landmark_id mapping
        df = pd.read_csv(meta_data_dir)
        self.label_map = dict(zip(df['id'], df['landmark_id']))
        
        self.embedding_list = []
        self.label_list = []
        for file_name in tqdm(os.listdir(self.dataset_preprocessed_dir)):
            image_id = os.path.splitext(file_name)[0]
            embeddings = torch.load(os.path.join(self.dataset_preprocessed_dir, file_name), map_location='cpu')
            # Keep on CPU, will move to GPU in __getitem__
            
            label = self.label_map[image_id]
            self.embedding_list.append(embeddings)
            self.label_list.append(label)
        
        self.embedding_dim = self.embedding_list[0]['cls_token'].shape[-1]
        
    def __len__(self):
        return len(self.embedding_list)
    
    def __getitem__(self, idx):
        embeddings = self.embedding_list[idx]
        label = self.label_list[idx]
        cls_token = embeddings['cls_token'].squeeze(0)
        patch_features = embeddings['patch_features'].squeeze(0)
        return cls_token, patch_features, label

def main(args):
    dataset_dir = args.dataset_dir
    device = args.device
    
    # train config
    hidden_dim = args.hidden_dim
    multihead_num = args.multihead_num
    layers_num = args.layers_num
    
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    
    dataset = Landmark_Dataset(dataset_dir, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    landmark_head = LandmarkHead(input_dim=dataset.embedding_dim, hidden_dim=hidden_dim, multihead_num=multihead_num, layers_num=layers_num, device=device)
    optimizer = torch.optim.AdamW(landmark_head.parameters(), lr=lr)
    
    # Contrastive learning: InfoNCE loss with temperature
    temperature = 0.5  # Increased for numerical stability
        
    for epoch in range(num_epochs):
        landmark_head.train()
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (cls_tokens, patch_features, labels) in enumerate(progress_bar):
            cls_tokens = cls_tokens.to(device)
            patch_features = patch_features.to(device)
            labels = labels.to(device)
            
            actual_batch_size = cls_tokens.size(0)  # Handle last batch
            
            # Forward pass: get embeddings
            embeddings = landmark_head(cls_tokens, patch_features)  # [batch_size, hidden_dim]
            
            # Normalize embeddings for contrastive learning
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Compute similarity matrix: [batch_size, batch_size]
            similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
            
            # Create contrastive labels: samples with same landmark_id are positives
            labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # [batch_size, batch_size]
            labels_eq.fill_diagonal_(False)  # Exclude self-similarity
            
            # InfoNCE loss: for each sample, pull positives and push negatives
            # Mask out the diagonal with large negative value instead of -inf
            mask = torch.eye(actual_batch_size, device=device, dtype=torch.bool)
            similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
            
            # For each anchor, compute loss against all samples
            losses = []
            
            for i in range(actual_batch_size):
                if labels_eq[i].sum() > 0:  # If there are positives in the batch
                    # Positive samples for anchor i
                    pos_mask = labels_eq[i]
                    # Compute log-sum-exp for all samples, then subtract log-sum-exp for positives
                    log_prob = F.log_softmax(similarity_matrix[i], dim=0)
                    sample_loss = -(log_prob * pos_mask.float()).sum() / pos_mask.sum()
                    losses.append(sample_loss)

            if len(losses) > 0:
                loss = torch.stack(losses).mean()
            else:
                # Skip batch if no valid pairs
                continue
            
            # Check for nan
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {batch_idx+1}, skipping...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(landmark_head.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            print(f"Step {batch_idx+1}, Loss: {loss.item():.4f}, Valid pairs: {len(losses)}")
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint (keep only last 5)
        if epoch % 20 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': landmark_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            # Remove old checkpoints, keep only last 5
            checkpoint_files = sorted(
                [f for f in os.listdir('.') if f.startswith('checkpoint_epoch_') and f.endswith('.pth')],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            while len(checkpoint_files) > 5:
                old_checkpoint = checkpoint_files.pop(0)
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
        
if __name__ == '__main__':
    args = arg_parse()
    main(args)