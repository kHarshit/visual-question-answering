from transformers import CLIPProcessor, CLIPModel
import torch

# Initialize CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embeddings(image_path, question):
    # Process image and text
    inputs = clip_processor(text=question, images=image_path, return_tensors="pt", padding=True)
    
    # Get the embeddings
    outputs = clip_model(**inputs)
    return outputs
