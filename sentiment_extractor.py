import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Load the CSV file
df = pd.read_csv("./data/Telco_customer_churn_with_text.csv")  # adjust the path accordingly

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# Apply the sentiment analysis function to the 'customer_text' column
df['customer_sentiment'] = df['customer_text'].apply(get_sentiment)