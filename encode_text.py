import sentence_transformers
import numpy as np
import pandas as pd

# Load the SentenceTransformer model
emb_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")

# Path to the original text file
input_file = 'fodor2.txt'
output_file = 'original_embedding2.tsv'

# Read the original text
with open(input_file, 'r', encoding='utf-8') as f:
    original_text = f.read().strip()

# Encode the text to a vector
original_embedding = emb_model.encode(original_text)

# Convert the embedding to a DataFrame and save to CSV
embedding_df = pd.DataFrame([original_embedding])
embedding_df.to_csv(output_file, index=False, header=False, sep='\t')

print(f"Original text embedding saved to {output_file}")
