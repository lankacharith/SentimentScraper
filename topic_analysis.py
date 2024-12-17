from sentiment_utils import initialize_sentiment_model, analyze_sentiment_percentage
from data_fetcher import search_google, extract_text_from_url

def analyze_topic(query, api_key, cx):
    sentiment_model = initialize_sentiment_model()
    urls = search_google(query, api_key, cx)

    total_positive, total_negative, total_sentences = 0, 0, 0

    for url in urls:
        article_text = extract_text_from_url(url)
        
        if article_text:
            results = analyze_sentiment_percentage(sentiment_model, article_text)

            total_positive += (results["positive_percentage"] * results["total_sentences"]) / 100
            total_negative += (results["negative_percentage"] * results["total_sentences"]) / 100
            total_sentences += results["total_sentences"]

    combined_positive_percentage = (total_positive / total_sentences) * 100
    combined_negative_percentage = (total_negative / total_sentences) * 100

    return combined_positive_percentage, combined_negative_percentage
