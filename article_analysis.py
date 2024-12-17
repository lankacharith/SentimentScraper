from sentiment_utils import initialize_sentiment_model, analyze_sentiment_percentage
from data_fetcher import extract_text_from_url

def analyze_single_article(url):
    sentiment_model = initialize_sentiment_model()
    article_text = extract_text_from_url(url)
    return analyze_sentiment_percentage(sentiment_model, article_text)