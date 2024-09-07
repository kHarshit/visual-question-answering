# Multimodal Visual Question Answering

Visual Question Answering System using ViT, GPT, BERT, CLIP (LLMs/VLMs)

Tasks and questions to ponder:
1. Late Fusion - process modalities separtely through Language or Vision models then merge their outputs
2. Early Fusion like Chameleon VLM (requires high computation power) - process modalities together after merging in the beginning itself.
3. (Fine-tuning) Final layer: Classification vs Generation layer (some design changes)
4. Can we utilize pre-trained VLM such as BLIP or MiniGPT (simiar to late fusion)?

![VQA page](./vqa_page.png)

## Models 

1. **BERT (Bidirectional Encoder Representations from Transformers)**: Encoder-only language model trained on masked language modeling (MLM) and next sentence prediction (NSP).
2. **ViT (Vision Transformer)**: Encoder-only vision model that treats image patches as tokens for efficient image processing.
3. **GPT-2 (Generative Pre-trained Transformer 2)**: Decoder-only language model designed for coherent text generation.
4. **CLIP (Contrastive Language-Image Pre-training)**: Model jointly trained an image and text encoder to understand and align visual and textual data in a shared latent space.

## Late Fusion Model (Classification)

[vqa_classification.ipynb](./vqa_classification.ipynb)

The late fusion model for Visual Question Answering (VQA) treats the task as a classification problem. It uses separate encoders for text and image inputs, which are fused together before making a classification prediction.

![late_fusion_classification](late_fusion_classification.png)

Architecture:

* Text Encoder: Pre-trained BERT model (bert-base-uncased).
* Image Encoder: Pre-trained Vision Transformer model (google/vit-base-patch16-224-in21k).
* Fusion Layer: Combines the outputs from the text and image encoders using a linear layer, followed by ReLU activation and dropout.
* Classifier: A linear layer that maps the fused representation to a fixed set of answer classes.


## Generation Model

[vqa_generation.ipynb](./vqa_generation.ipynb)

The generation model treats VQA as a sequence generation problem. It integrates separate encoders for text and image inputs and uses a decoder to generate textual answers.

![late_fusion_generation](late_fusion_generation.png)
Architecture:

* Text Encoder: Pre-trained BERT model (bert-base-uncased).
* Image Encoder: Pre-trained Vision Transformer model (google/vit-base-patch16-224-in21k).
* Fusion Layer: Combines the outputs from the text and image encoders using a linear layer, followed by ReLU activation and dropout.
* Text Decoder: Pre-trained GPT-2 model (gpt2) for generating the answer text.

## Running

Create conda env

```
conda env create -f environment.yml
```


Using docker

```
# run app
python app.py

# use docker
transformers-cli serve --task=fill-mask --model=bert-base-uncased

curl -X POST http://localhost:8888/forward -H "accept: application/json" -H "Content-Type: application/json" -d '{"inputs": "Today is going to be a [MASK] day"}' | jq

docker build --platform linux/amd64 -t vqa:v1 .
# check port from docker ps and use the curl command to get output
```