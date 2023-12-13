from flask import Flask, request, render_template
import base64
from PIL import Image
from io import BytesIO
from transformers import pipeline

app = Flask(__name__)

vqa_pipeline = pipeline("visual-question-answering")

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
