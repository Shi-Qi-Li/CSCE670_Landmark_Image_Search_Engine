import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from torchvision.models.vision_transformer import Encoder

class ImageEncoder:
    def __init__(self, lib='facebook/dinov3-vit7b16-pretrain-lvd1689m', device='cpu'):
        self.processor = AutoImageProcessor.from_pretrained(lib)
        self.model = AutoModel.from_pretrained(lib).to(device)
        self.device = device
    
    def encode(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        cls_token = last_hidden_states[:, 0, :]
        patch_features_flat = last_hidden_states[:, 1 + self.model.config.num_register_tokens:, :]
        
        return {'cls_token': cls_token, 'patch_features': patch_features_flat}