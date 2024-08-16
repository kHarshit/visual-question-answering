from flask import Flask, request, render_template
import base64
from PIL import Image
from io import BytesIO
from transformers import pipeline

app = Flask(__name__)

vqa_pipeline = pipeline("visual-question-answering")

inference_script = """
# testing this requires gpu and trained weights.
#!/usr/bin/env python
# coding: utf-8

import os
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the tokenizer and model

class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_labels: int = len(answer_space),
            intermediate_dim: int = 512,
            pretrained_text_name: str = 'bert-base-uncased',
            pretrained_image_name: str = 'google/vit-base-patch16-224-in21k'):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        
        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text['pooler_output'],
                    encoded_image['pooler_output'],
                ],
                dim=1
            )
        )
        logits = self.classifier(fused_output)
        
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out

model = MultimodalVQAModel()
# We use the checkpoint giving best results
model.load_state_dict(torch.load(os.path.join("..", "checkpoint", "bert_vit", "checkpoint-1500", "pytorch_model.bin")))
model.to(device) 
model.eval()

def preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    # Include any image preprocessing here
    return image

def preprocess_text(question):
    # Tokenize the question
    inputs = tokenizer(question, return_tensors="pt")
    return inputs

def predict(image_path, question):
    # Preprocess inputs
    image = preprocess_image(image_path)
    inputs = preprocess_text(question)

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process outputs
    # (Add your own logic here to interpret model outputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)

    return prediction

if __name__ == "__main__":
    # Example usage
    image_path = "path_to_image.jpg"
    question = "What is in the image?"
    prediction = predict(image_path, question)
    print(f"Prediction: {prediction}")
"""

def convert_image_to_base64(image):
    pil_img = Image.open(image)
    # Convert RGBA to RGB if the image has an alpha channel
    if pil_img.mode == 'RGBA':
        pil_img = pil_img.convert('RGB')

    img_buffer = BytesIO()
    pil_img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)

    image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ""
    image_data = None

    if request.method == 'POST':
        image = request.files['image']
        question = request.form['question']

        image_data = convert_image_to_base64(image)

        # Process the image and question to get an answer
        answer = process_vqa(image, question)
    
    return render_template('index.html', answer=answer, question=question, image_data=image_data)

@app.route('/answer', methods=['POST'])
def answer():
    image = request.files['image']
    question = request.form['question']

    # Process the image and question to get an answer (to be implemented)
    answer = process_vqa(image, question)
    
    return render_template('result.html', answer=answer)  # Create a result.html file in templates folder

def process_vqa(image, question):
    # Implement the function to process the VQA using the CLIP embeddings
    # Placeholder return
    # Use Hugging Face VQA pipeline
    pil_img = Image.open(image)
    result = vqa_pipeline(pil_img, question, top_k=1)
    return result[0]['answer']
    # return "This is a sample answer."

if __name__ == '__main__':
    app.run(debug=True)
