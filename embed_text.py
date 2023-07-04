from transformers import BertModel, BertTokenizer
import torch
import numpy as np

# Use a pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    # Tokenize the text and convert it to a tensor
    inputs = tokenizer(text, return_tensors='pt')

    # Generate an embedding for the text
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the average of the last hidden state as the embedding
    embedding = outputs.last_hidden_state.mean(dim=1)
    
    # Convert the tensor to a numpy array and write it to a file
    np.save('embedding.npy', embedding.numpy())

embed_text("Some example text.")