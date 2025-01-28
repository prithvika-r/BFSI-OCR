import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Load description dataset
df = pd.read_csv(r"C:\Users\prithvika\OneDrive\Desktop\springboard\mccmodel\extracted_mcc_codes.csv", dtype=str)
df.columns = df.columns.str.strip()  # Clean column names
df["MCC Code"] = df["MCC Code"].str.strip()  # Strip spaces in MCC codes

# Generate embeddings for the description dataset
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

df["embedding"] = df["Description"].apply(lambda x: get_embedding(x).tolist())

# Search MCC Code or Description
def find_best_match(mcc_code, description=None):
    mcc_code = mcc_code.strip()  # Remove extra spaces
    if mcc_code.isdigit():
        match = df[df["MCC Code"] == mcc_code]
        if not match.empty:
            return match["Description"].values[0]
        else:
            return "MCC code not found."
    elif description:
        query_embedding = get_embedding(description)
        embeddings = np.array(df["embedding"].tolist())
        similarities = cosine_similarity([query_embedding], embeddings)
        most_similar_index = similarities.argmax()
        return df.iloc[most_similar_index]["Description"]
    return "Invalid input."

# Process input CSV
def process_input_csv(input_csv_path, output_csv_path):
    # Load the input CSV
    input_df = pd.read_csv(input_csv_path, dtype=str)
    input_df.columns = input_df.columns.str.strip()

    # Split 'Amount,MCC Code' into two separate columns if it exists
    if "Amount,MCC Code" in input_df.columns:
        input_df[["Amount", "MCC Code"]] = input_df["Amount,MCC Code"].str.split(",", expand=True)
        input_df["MCC Code"] = input_df["MCC Code"].str.strip()  # Ensure MCC Code is stripped of spaces
        input_df.drop("Amount,MCC Code", axis=1, inplace=True)

    # Check for required column
    if "MCC Code" not in input_df.columns:
        raise ValueError("The input CSV must contain an 'MCC Code' column.")

    # Add descriptions based on MCC Code
    input_df["Predicted Description"] = input_df["MCC Code"].apply(find_best_match)

    # Save the processed CSV
    input_df.to_csv(output_csv_path, index=False)
    print(f"Processed file saved at: {output_csv_path}")

# Paths to input and output files
input_csv_path = "mccnocat.csv"
output_csv_path = "test.csv"

# Process the input CSV
process_input_csv(input_csv_path, output_csv_path)
