#!/usr/bin/env python
# coding: utf-8

import os
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import unittest

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")
model = AutoModel.from_pretrained("MODEL_NAME")
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
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)

    return prediction

# Unit tests
class TestInferenceMethods(unittest.TestCase):

    def test_preprocess_image(self):
        # Test that image preprocessing does not fail
        image = preprocess_image("image001.jpg")
        self.assertIsInstance(image, Image.Image)

    def test_preprocess_text(self):
        # Test that text preprocessing returns the correct structure
        inputs = preprocess_text("What is this?")
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertEqual(inputs["input_ids"].shape[1], len("What is this?"))

    def test_predict(self):
        # Test that prediction returns a valid tensor
        prediction = predict("path_to_test_image.jpg", "What is in the image?")
        self.assertIsInstance(prediction, torch.Tensor)
        self.assertEqual(len(prediction.shape), 1)  # Ensure it's a 1D tensor

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False)

    # Example usage
    image_path = "image001.jpg"
    question = "What is in the image?"
    prediction = predict(image_path, question)
    print(f"Prediction: {prediction}")
