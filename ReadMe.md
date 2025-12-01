# Landmark Retrieval

## Install

```bash
conda env create -f environment.yml
conda activate landmark-search
```

## Module

### DINOv3 Encoder

The default encoder library is [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m). You can find more DINOv3 encoder libraries in [DINOv3](https://huggingface.co/collections/facebook/dinov3). **The Output Head is trained on the default encoder, and is not compaitable with other modek version.**

```python
from image_encoder import ImageEncoder

image_encoder = ImageEncoder(lib='facebook/dinov3-vits16-pretrain-lvd1689m', device='cuda:0')

# forward
from PIL import Image
image_dir = '' # path of image
image = Image.open('image_dir').convert('RGB')
embeddings = image_encoder([image])

cls_token = embeddings['cls_token']
patch_features = embeddings['patch_features']
```

### Output Head

The **Output Head** is trained for encode the embeddings from DINOv3 into feature vector with 512 dimension.

```python
from landmark_head import LandmarkHead
from PIL import Image
from image_encoder import ImageEncoder

# load output head
landmark_head = LandmarkHead(input_dim=384, hidden_dim=512, multihead_num=multihead_num, layers_num=layers_num, device=device)
# remember to load the checkpoint

# load image encoder
image_encoder = ImageEncoder(lib='facebook/dinov3-vits16-pretrain-lvd1689m', device='cuda:0')

# encoding image
image_dir = '' # path of image
image = Image.open('image_dir').convert('RGB')
embeddings = image_encoder([image])

cls_token = embeddings['cls_token']
patch_features = embeddings['patch_features']

# inference feature vector
vector = landmark_head(cls_tokens, patch_features) # dim = 512

```

### Dataset

The dataset is located in /data/hkzhang/GLDv2. I have authorize the access permission. It includes 90k images. The image is stored in /data/hkzhang/GLDv2/0, metadata is stored in /data/hkzhang/GLDv2/train.csv.

## Gradio Demo

Use DINOv3 cls_token for coarse retrieval.
```
python app.py --dataset_dir your_dataset_path --mode cls
```
Use landmark head feature for coarse retrieval.
```
python app.py --dataset_dir your_dataset_path --ckpt your_ckpt_path
```