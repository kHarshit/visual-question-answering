FROM huggingface/transformers-pytorch-gpu

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN pip install fastapi uvicorn

RUN python3 -c "from transformers import AutoModel;AutoModel.from_pretrained('bert-base-uncased')"
RUN python3 -c "from transformers import AutoTokenizer;AutoTokenizer.from_pretrained('bert-base-uncased')"

EXPOSE 8888
ENTRYPOINT [ "transformers-cli", "serve", "--port=8888", "--host=0.0.0.0", "--task=fill-mask", "--model=bert-base-uncased"]

