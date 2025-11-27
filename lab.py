import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from landmark_head import LandmarkHead
from image_encoder import ImageEncoder


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
print("Image size:", image.height, image.width)  # [480, 640]

device = 'cuda:1'
image_encoder = ImageEncoder(device=device)
landmark_head = LandmarkHead(input_dim=4096, hidden_dim=512, multihead_num=4, layers_num=3, device=device)

encoder_outputs = image_encoder.encode([image])
cls_token = encoder_outputs['cls_token']
patch_features = encoder_outputs['patch_features']
landmark_outputs = landmark_head(cls_token, patch_features)
print("Landmark output shape:", landmark_outputs.shape)