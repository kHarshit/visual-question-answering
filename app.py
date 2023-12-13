from flask import Flask, request, render_template
import base64
from io import BytesIO

app = Flask(__name__)

def convert_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
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
    return "This is a sample answer."

if __name__ == '__main__':
    app.run(debug=True)
