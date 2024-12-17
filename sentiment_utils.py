from transformers import pipeline
from nltk.tokenize import sent_tokenize

def initialize_sentiment_model():
    return pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english", device = 0)

def analyze_sentiment_percentage(sentiment_model, text):
    if not text:
        return "No text found to analyze."
    
    sentences = sent_tokenize(text)

    sentiment_results = [sentiment_model(sentence[:512])[0] for sentence in sentences]  # Truncate sentences (model limitation 512 tokens)

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