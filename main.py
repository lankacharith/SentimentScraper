import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from torch.utils.data import DataLoader

API_KEY = 'AIzaSyDwpTxBMH99xAMc7gcHG2jYH4arUiovbZU'
CX = 'a6112534b06a64b0f'

model = None
tokenizer = None

dataset = load_dataset("glue", "sst2")

# Load model and tokenizer
def initialize_sentiment_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        model_name = "google/bigbird-roberta-base"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2) # 2 labels for sentiment (positive/negative)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

model, tokenizer = initialize_sentiment_model()

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)    

# Training Dataset
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

train_dataloader = DataLoader(train_dataset, batch_size=8)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    evaluation_strategy="epoch",     # Evaluation frequency
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Weight decay for regularization
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

def search_google(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX}"
    response = requests.get(search_url)
    data = response.json()

    if 'items' in data:
        return [item['link'] for item in data['items']]

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the main content of the article (may vary by site)
        paragraphs = soup.find_all('p')
        text = ' '.join(paragraph.get_text() for paragraph in paragraphs)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

def analyze_sentiment(sentiment_model, text):
    if not text:
        return "No text found to analyze."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    tokenized_text = inputs['input_ids']

    chunk_size = 4096  # Adjust based on model
    num_chunks = (len(tokenized_text[0]) // chunk_size) + 1

    sentiment_scores = []
    
    for i in range(num_chunks):
        chunk = tokenized_text[0][i * chunk_size:(i + 1) * chunk_size]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        results = sentiment_model(chunk_text)
        sentiment_scores.extend(results)

    positive_score = sum(1 for result in sentiment_scores if result['label'] == 'POSITIVE')
    negative_score = sum(1 for result in sentiment_scores if result['label'] == 'NEGATIVE')

    overall_sentiment = "POSITIVE" if positive_score > negative_score else "NEGATIVE"
    return {
        "overall_sentiment": overall_sentiment,
        "positive_chunks": positive_score,
        "negative_chunks": negative_score,
        "total_chunks": len(sentiment_scores),
    }

def analyze_sentiment_percentage(sentiment_model, text):
    if not text:
        return "No text found to analyze"
    
    sentences = sent_tokenize(text)

    sentiment_results = [sentiment_model(sentence)[0] for sentence in sentences]

    positive_count = sum(1 for result in sentiment_results if result['label'] == 'POSITIVE')
    negative_count = sum(1 for result in sentiment_results if result['label'] == 'NEGATIVE')
    total_sentences = len(sentiment_results)

    positive_percentage = (positive_count / total_sentences) * 100
    negative_percentage = (negative_count / total_sentences) * 100
    # neutral_percentage = 100 - (positive_percentage - negative_percentage)

    if positive_percentage > negative_percentage:
        overall_sentiment = "POSITIVE"
    elif negative_percentage > positive_percentage:
        overall_sentiment = "NEGATIVE"
    else:
        overall_sentiment = "NEUTRAL"

    return {
        "overall_sentiment": overall_sentiment,
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage,
        # "neutral_percentage": neutral_percentage,
        "total_sentences": total_sentences,
    }

def main():
    query = input("Enter a topic to analyze: ")
    print(f"\nFetching information related to {query}...")
    
    urls = search_google(query)

    if not urls:
        print("No information found.")
        return

    sentiment_model = initialize_sentiment_model()

    total_positive, total_negative, total_sentences = 0, 0, 0
    
    for url in urls:
        print(f"\nAnalyzing {url}...")
        article_text = extract_text_from_url(url)

        if article_text:
            sentiment_results = analyze_sentiment_percentage(sentiment_model, article_text)
            # print(f"Overall Sentiment: {sentiment_results['overall_sentiment']}")
            # print(f"Positive Sentiment: {sentiment_results['positive_percentage']:.2f}%")
            # print(f"Negative Sentiment: {sentiment_results['negative_percentage']:.2f}%")
            # print(f"Neutral Sentiment: {sentiment_results['neutral_percentage']:.2f}%")

            total_positive += (sentiment_results["positive_percentage"] * sentiment_results["total_sentences"]) / 100
            total_negative += (sentiment_results["negative_percentage"] * sentiment_results["total_sentences"]) / 100
            total_sentences += sentiment_results["total_sentences"]
        else:
            print("Failed to extract text from this website.")
    
    combined_positive_percentage = (total_positive / total_sentences) * 100
    combined_negative_percentage = (total_negative / total_sentences) * 100

    if combined_positive_percentage > combined_negative_percentage:
        combined_sentiment = "POSITIVE"
    elif combined_negative_percentage > combined_positive_percentage:
        combined_sentiment = "NEGATIVE"
    else:
        combined_sentiment = "NEUTRAL"

    print(f"\nOverall Sentiment: {combined_sentiment}")
    print(f"Positive Sentiment: {combined_positive_percentage:.2f}%")
    print(f"Negative Sentiment: {combined_negative_percentage:.2f}%")

if __name__ == "__main__":
    main()