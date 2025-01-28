"""
Transaction Classification using BERT

- Loads a synthetic dataset of transaction descriptions and categories.
- Preprocesses and encodes the dataset with LabelEncoder.
- Splits data into training and testing sets (80-20 split).
- Tokenizes descriptions using the BERT tokenizer.
- Defines a custom PyTorch dataset class for transaction data.
- Loads the pre-trained BERT model for sequence classification.
- Trains the model for 10 epochs using AdamW optimizer.
- Evaluates the model on the test dataset to measure accuracy.
- Saves the trained model and tokenizer for future use.
- Provides a prediction function to classify transaction descriptions into categories.
- Allows user interaction for live predictions by entering transaction descriptions.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r'C:\Users\prithvika\OneDrive\Desktop\springboard\Synthetic_Data.csv')

# Check and clean column names 
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces
print("Columns in the dataset:", df.columns)
df.rename(columns={'Description': 'Description', 'Category': 'Category'}, inplace=True)
# Check for missing values and remove
df.dropna(subset=['Description', 'Category'], inplace=True)

#encode categories to numerical values
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])

#split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define PyTorch dataset
class TransactionDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_length=64):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = str(self.descriptions[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

#Create DataLoader
train_dataset = TransactionDataset(train_df['Description'].values, train_df['Category'].values, tokenizer)
test_dataset = TransactionDataset(test_df['Description'].values, test_df['Category'].values, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

#load BERT model for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

#set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

#CPU device is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Training
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

#Evaluation
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

#Train the model
epochs = 10
for epoch in range(epochs): 
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_dataloader, optimizer)
    accuracy = evaluate(model, test_dataloader)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

#save the trained model
model.save_pretrained(r'C:\Users\prithvika\OneDrive\Desktop\springboard\transaction_classifier')
tokenizer.save_pretrained(r'C:\Users\prithvika\OneDrive\Desktop\springboard\transaction_classifier')

# Function to predict category for a given description
def predict_category(description):
    model.eval()
    encoding = tokenizer.encode_plus(
        description,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)

    predicted_label = label_encoder.inverse_transform([predicted.item()])
    return predicted_label[0]

# Testing the model
while True:
    user_input = input("\nEnter a transaction description (or type 'exit' to quit): ").strip()
    
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break  # Exit the loop
    
    predicted_category = predict_category(user_input)
    print(f"Predicted Category for '{user_input}': {predicted_category}")
