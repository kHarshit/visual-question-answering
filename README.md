# visual-question-answering
Visual Question Answering System using ViT, GPT, BERT (LLMs)

Tasks and questions to ponder:
1. Late Fusion - process modalities separtely through Language or Vision models then merge their outputs
2. Early Fusion like Chameleon VLM (requires high computation power) - process modalities together after merging in the beginning itself.
3. (Fine-tuning) Final layer: Classification vs Generation layer (some design changes)
4. Can we utilize pre-trained VLM such as BLIP or MiniGPT (simiar to late fusion)?

![VQA page](./vqa_page.png)

## Late Fusion Model (Classification)

The late fusion model for Visual Question Answering (VQA) treats the task as a classification problem. It uses separate encoders for text and image inputs, which are fused together before making a classification prediction.

Architecture:

* Text Encoder: Pre-trained BERT model (bert-base-uncased).
* Image Encoder: Pre-trained Vision Transformer model (google/vit-base-patch16-224-in21k).
* Fusion Layer: Combines the outputs from the text and image encoders using a linear layer, followed by ReLU activation and dropout.
* Classifier: A linear layer that maps the fused representation to a fixed set of answer classes.


## Generation Model

The generation model treats VQA as a sequence generation problem. It integrates separate encoders for text and image inputs and uses a decoder to generate textual answers.

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