import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import warnings
import gc
from sklearn.metrics import precision_recall_fscore_support

# Ignore warnings
warnings.filterwarnings('ignore')

# Clean up memory
gc.collect()

# Create a directory to save the model
os.makedirs('model', exist_ok=True)

# Load the data
df = pd.read_csv('data/train.tsv', sep="\t", usecols=['Phrase', 'Sentiment'])


# Define the mapping function
def map_sentiment(value):
    if value in [0, 1]:
        return 0
    elif value == 2:
        return 1
    else:
        return 2


# Apply the mapping
df['Sentiment'] = df['Sentiment'].apply(map_sentiment)

# Split the training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['Phrase'].values, df['Sentiment'].values,
                                                                      test_size=0.2, random_state=42)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')

# Tokenize the text
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Create the dataset
train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)
test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(test_labels)
)

# Create the data loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the pre-trained model
model = BertForSequenceClassification.from_pretrained('bert_base_uncased', num_labels=3)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train the model
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        total += labels.size(0)

        # Update the progress bar
        progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1), 'accuracy': correct / total})
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print(
        f"Train epoch:{epoch} Accuracy: {correct / total:.4f} Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f}")
# Evaluate the model on the test set
model.eval()
test_correct = 0
test_total = 0
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.4f} Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f}')

# Save the model and tokenizer
model.save_pretrained('model/bert_sentiment_3')
tokenizer.save_pretrained('model/bert_sentiment_3')

print("Model and tokenizer saved to 'model/bert_sentiment_3'")
