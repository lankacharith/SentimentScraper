import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def initialize_sentiment_model():
    return pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english", device = 0)

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

def main():
    url = input("Enter the website URL: ")
    print("Fetching and analyzing article...")

    sentiment_model = initialize_sentiment_model()
    article_text = extract_text_from_url(url)

    if article_text:
        sentiment_results = analyze_sentiment(sentiment_model, article_text)
        print(f"\nAnalysis Results:")
        print(f"Overall Sentiment: {sentiment_results['overall_sentiment']}")
        print(f"Positive Chunks: {sentiment_results['positive_chunks']} / {sentiment_results['total_chunks']}")
        print(f"Negative Chunks: {sentiment_results['negative_chunks']} / {sentiment_results['total_chunks']}")
    else:
        print("Failed to extract text from the article.")

if __name__ == "__main__":
    main()