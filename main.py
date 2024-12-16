import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from nltk.tokenize import sent_tokenize

API_KEY = 'AIzaSyDwpTxBMH99xAMc7gcHG2jYH4arUiovbZU'
CX = 'a6112534b06a64b0f'

def initialize_sentiment_model():
    return pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english", device = 0)

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

        paragraphs = soup.find_all('p')
        text = ' '.join(paragraph.get_text() for paragraph in paragraphs)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

def analyze_sentiment(sentiment_model, text):
    if not text:
        return "No text found to analyze."

    chunk_size = 512  # Adjust based on model
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sentiment_scores = []

    for chunk in text_chunks:
        results = sentiment_model(chunk)
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