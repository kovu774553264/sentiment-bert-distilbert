import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load the saved model and tokenizer
model_path = 'model/distilbert_sentiment'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()

# Define sentiment labels
sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']


def predict_sentiment(text):
    # Encode the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Obtain the predicted sentiment label
    predicted_sentiment = sentiment_labels[predicted_class]

    return predicted_sentiment


# Main program
if __name__ == "__main__":
    while True:
        # Get user input
        user_input = input("Enter an English sentence (or 'quit' to exit): ")

        if user_input.lower() == 'quit':
            break

        # Predict sentiment
        sentiment = predict_sentiment(user_input)
        print(f"Predicted sentiment: {sentiment}")
        print()

print("Thank you for using the sentiment analysis tool!")
