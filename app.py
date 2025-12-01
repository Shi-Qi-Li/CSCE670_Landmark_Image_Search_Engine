import os
import torch
import random
import argparse
import gradio as gr
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from image_encoder import ImageEncoder
from landmark_head import LandmarkHead

# Set custom temp directory for Gradio
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), 'gradio_tmp')
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)


def arg_parse():
    parser = argparse.ArgumentParser(description="Landmark Head Demo")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", choices=["cls", "landmark"], default="landmark", help="Coarse search mode")
    parser.add_argument("--num_candidates", type=int, default=20, help="Number of candidate images to display for selection")
    return parser.parse_args()


def load_data(dataset_dir):
    dataset_preprocessed_dir = os.path.join(dataset_dir, "preprocessed")
    meta_data_dir = os.path.join(dataset_dir, "train.csv")
    
    # Load metadata CSV to create image_id -> landmark_id mapping
    df = pd.read_csv(meta_data_dir)
    label_map = dict(zip(df["id"], df["landmark_id"]))
    
    embedding_list = []
    label_list = []
    id_list = []
    for file_name in tqdm(os.listdir(dataset_preprocessed_dir)):
        image_id = os.path.splitext(file_name)[0]
        embeddings = torch.load(os.path.join(dataset_preprocessed_dir, file_name), map_location="cpu")
        
        label = label_map[image_id]
        embedding_list.append(embeddings)
        label_list.append(label)
        id_list.append(image_id)

    return embedding_list, label_list, id_list


def search_similar_images(query_glo_embedding, query_patch_feature, database_glo_embeddings, database_patch_features, top_k=5, coarse_top_k=200):
    """
    Perform a two-stage image similarity search.
    """
    # Compute distances between query and database CLS embeddings
    cls_distances = torch.cdist(query_glo_embedding.unsqueeze(0), database_glo_embeddings).squeeze(0)
    
    # Get top-k closest images based on CLS embeddings
    topk_values, topk_indices = torch.topk(cls_distances, k=coarse_top_k+1, largest=False, sorted=True)
    topk_values = topk_values[1:]  # Exclude the query image itself
    topk_indices = topk_indices[1:]  # Exclude the query image itself
    
    # Refine search using patch features
    query_patch_feature = query_patch_feature.unsqueeze(0)  # Shape: (1, N, D)
    candidate_patch_features = database_patch_features[topk_indices]  # Shape: (coarse_top_k, N, D)
    
    # The patch feature distance can be considered as the Chamfer distance between two patch sets
    per_patch_pair_distances = torch.norm(query_patch_feature.unsqueeze(-2) - candidate_patch_features.unsqueeze(-3), dim=-1)
    per_patch_nearest_distances, _ = torch.min(per_patch_pair_distances, dim=-1)
    refined_distances = torch.mean(per_patch_nearest_distances, dim=-1)
    
    topk_values, refined_topk_indices = torch.topk(refined_distances, k=top_k, largest=False, sorted=True)
    topk_indices = topk_indices[refined_topk_indices]

    return topk_indices


def process_raw_image(image_path, args):
    """Load and preprocess the image from the given path."""
    image = Image.open(image_path).convert("RGB")
    # load image encoder
    image_encoder = ImageEncoder(lib=args.dino_lib, device=args.device)
    embeddings = image_encoder.encode([image])

    # model config
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    multihead_num = args.multihead_num
    layers_num = args.layers_num
    
    landmark_head = LandmarkHead(input_dim=input_dim, hidden_dim=hidden_dim, multihead_num=multihead_num, layers_num=layers_num, device=args.device)
    landmark_head.load_state_dict(torch.load(args.ckpt, map_location=args.device)["model_state_dict"])
    landmark_head.eval()

    cls_tokens = embeddings["cls_token"]  # [1, input_dim]
    patch_features = embeddings["patch_features"]  # [1, num_patches, input_dim]
    
    with torch.no_grad():
        retrieval_feature = landmark_head(cls_tokens, patch_features)  # [1, hidden_dim]
    retrieval_feature = F.normalize(retrieval_feature, p=2, dim=1)

    glo_embeddings = retrieval_feature.squeeze_(0) if args.mode == "landmark" else cls_tokens.squeeze_(0)
    patch_features = patch_features.squeeze_(0)

    return glo_embeddings, patch_features


def get_image_path(image_id, dataset_dir):
    """Construct the full path to an image given its ID."""
    return os.path.join(dataset_dir, "train", image_id[0], image_id[1], image_id[2], f"{image_id}.jpg")


def create_gui(args, id_list, glo_embeddings, patch_features):
    """Create and launch the Gradio interface."""
    
    num_samples = min(args.num_candidates, len(id_list))
    candidate_indices = []
    candidate_images_cache = []  # Store loaded images
    
    def sample_new_candidates():
        """Sample new random candidates."""
        nonlocal candidate_indices, candidate_images_cache
        candidate_indices = random.sample(range(len(id_list)), num_samples)
        candidate_images_cache = []
    
    def load_candidate_images():
        """Load candidate images for user selection."""
        candidates = []
        candidate_images_cache.clear()
        for idx in candidate_indices:
            img_path = get_image_path(id_list[idx], args.dataset_dir)
            try:
                img = Image.open(img_path).convert("RGB")
                candidate_images_cache.append((idx, img))  # Cache the images
                # candidates.append((img, f"ID: {id_list[idx]}\nLabel: {label_list[idx]}"))
                candidates.append((img, f"ID: {id_list[idx]}"))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        return candidates
    
    def refresh_candidates():
        """Refresh candidate images with a new batch."""
        sample_new_candidates()
        return load_candidate_images()
    
    # Initial sampling
    sample_new_candidates()
    
    def perform_search(evt: gr.SelectData):
        """Perform image search when user selects a query image."""
        # Get the selected image index from candidate_indices
        selected_idx = candidate_indices[evt.index]
        
        # Perform search
        search_result = search_similar_images(
            query_glo_embedding=glo_embeddings[selected_idx],
            query_patch_feature=patch_features[selected_idx],
            database_glo_embeddings=glo_embeddings,
            database_patch_features=patch_features,
            top_k=5,
            coarse_top_k=200
        )
        
        # Load result images
        result_images = []
        for rank, idx in enumerate(search_result.tolist(), 1):
            result_img_path = get_image_path(id_list[idx], args.dataset_dir)
            try:
                result_img = Image.open(result_img_path).convert("RGB")
                # result_images.append((result_img, f"Rank {rank}\nID: {id_list[idx]}\nLabel: {label_list[idx]}"))
                result_images.append((result_img, f"Rank {rank}\nID: {id_list[idx]}"))
            except Exception as e:
                print(f"Error loading result image {result_img_path}: {e}")
        
        return result_images
    
    # Create Gradio interface
    with gr.Blocks(title="Image Retrieval System") as demo:
        gr.Markdown("# 🔍 Image Retrieval System")
        gr.Markdown("Select a query image from the candidates on the left, then click on it to see similar images on the right.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Query Candidates")
                with gr.Row():
                    gr.Markdown("Click on any image to use it as a query")
                    refresh_btn = gr.Button("🔄 Load New Batch", size="sm")
                candidate_gallery = gr.Gallery(
                    label="Select Query Image",
                    columns=4,
                    rows=5,
                    height="auto",
                    object_fit="contain"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Search Results (Top 5)")
                result_gallery = gr.Gallery(
                    label="Similar Images",
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )
        
        # Set up events
        candidate_gallery.select(perform_search, None, result_gallery)
        refresh_btn.click(refresh_candidates, None, candidate_gallery)
        
        # Load candidate images after interface is created
        demo.load(load_candidate_images, None, candidate_gallery)
    
    return demo


def main(args):
    print("Loading data...")
    embedding_list, label_list, id_list = load_data(args.dataset_dir)
    if args.mode == "cls":
        glo_embeddings = torch.concat([emb["cls_token"] for emb in embedding_list]).to(args.device)
    else:
        glo_embeddings = torch.concat([emb["retrieval_feature"] for emb in embedding_list]).to(args.device)
    patch_features = torch.concat([emb["patch_features"] for emb in embedding_list]).to(args.device)
    
    print(f"Loaded {len(id_list)} images from the database.")
    print("Creating GUI...")
    
    demo = create_gui(args, id_list, glo_embeddings, patch_features)
    demo.launch(share=False)


if __name__ == "__main__":
    args = arg_parse()
    main(args)