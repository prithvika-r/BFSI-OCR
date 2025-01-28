from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=11)  # Adjust num_labels for 11 categories

# Step 3: Define category keywords
category_keywords = {
    'Shopping': ['store', 'shop','sales','purchase', 'buy', 'mall', 'retail', 'style', 'flipkart', 'amazon'],
    'Groceries': ['fruits','super', 'groceries', 'vegetables', 'monthly provisions', 'provisions', 'bigbasket', 'supermarket'],
    'Transport': ['bus', 'train', 'flight', 'car', 'ride', 'taxi', 'uber', 'travel','vehicle','fuel'],
    'Food': ['restaurant', 'food', 'meal', 'coffee', 'cafe', 'fastfood','hotel'],
    'Bills': ['utility', 'bill', 'payment', 'electricity', 'water', 'phone', 'gas', 'internet', 'rent', 'petrol', 'diesel','station','airtel', 'bsnl'],
    'ATM': ['atm', 'cash withdrawal', 'withdrawal'],
    'Entertainment': ['movie', 'subscription', 'concert', 'ott', 'music', 'event', 'theater', 'tickets', 'game', 'streaming', 'entertainment'],
    'Medical': ['hospital', 'doctor', 'clinic', 'medicine', 'pharmacy', 'Healthcare', 'medical','drug'],
    'Savings': ['savings', 'investment', 'saving', 'account', 'interest', 'nettxn'],
    'Funds Transfer': ['UPI', 'Payee', 'Mutualfunds', 'deposit', 'IMPS', 'RTGS', 'cashdep', 'ifsc', 'paytm', 'googlepay', 'emi', 'neft', 'nett', 'bank', 'funds'],
    'Other': []
}

# Step 4: Preprocess the transaction data
def preprocess_data(text):
    if not isinstance(text, str):
        text = str(text)  # Convert non-string values to string (e.g., NaN or numbers)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs

# Step 5: Keyword-based categorization
def rule_based_classification(text):
    if not isinstance(text, str) or text.strip() == '':  # Handle empty or non-string text
        return 'Unknown'
    text_lower = text.lower()
    for category, keywords in category_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return 'Unknown'

# Step 6: Combine BERT with Rule-based classification
def classify_transaction(text):
    if not isinstance(text, str) or text.strip() == '' or text == '0' or text == '0.0':
        return 'Unknown'
    category = rule_based_classification(text)
    if category == 'Unknown':
        inputs = preprocess_data(text)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        categories = {
            0: 'Shopping',
            1: 'Groceries',
            2: 'Transport',
            3: 'Food',
            4: 'Bills',
            5: 'ATM',
            6: 'Entertainment',
            7: 'Medical',
            8: 'Savings',
            9: 'Unknown',
            10: 'Other'
        }
        return categories.get(predicted_class, 'Unknown')
    return category

# Step 7: Process a dataset and save the output with category columns
def classify_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    if 'Description' not in df.columns:
        print("Error: 'Description' column not found in dataset")
        return
    df['predicted_category'] = df['Description'].apply(classify_transaction)
    df.to_csv(output_file, index=False)
    print(f"Classification completed. Results saved to {output_file}")

    # Visualization: Bar chart
    plt.figure(figsize=(10, 6))
    category_counts = df['predicted_category'].value_counts()
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    plt.title('Distribution of Transaction Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    plt.show()

    # Visualization: Pie chart
    colors = sns.color_palette("pastel", len(category_counts))  # Softer pastel colors
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        category_counts.values, 
        labels=category_counts.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors, 
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},  # Add a subtle border for clarity
        pctdistance=0.85,  # Move percentage labels slightly away from the center
        labeldistance=1.2   # Move category labels further from the center
    )

    # Adjust the label text font and color
    for text in texts:
        text.set_fontsize(12)  # Set font size for category labels
        text.set_color('black')

    # Adjust the auto percentage text font and color
    for autotext in autotexts:
        autotext.set_fontsize(10)  # Set font size for percentage labels
        autotext.set_color('black')

    ax.set_title('Transaction Categories Distribution (Pie Chart)', fontsize=14)
    plt.tight_layout()
    plt.savefig('category_distribution_pie.png')
    plt.show()

# Example usage
input_file = 'bank_transactions.csv'
output_file = 'classified_transactions12.csv'
classify_dataset(input_file, output_file)
