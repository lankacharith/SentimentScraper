from topic_analysis import analyze_topic
from article_analysis import analyze_single_article
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
CX = os.getenv("CX")

def main():
    print("\nChoose an option:")
    print("1. Analyze a single article")
    print("2. Analyze a topic")

    choice = input("\nEnter 1 or 2: ")

    if choice == '1':
        url = input("\nEnter the URL of the article: ")
        print(f"\nAnalyzing {url}...")
        
        try:
            sentiment_results = analyze_single_article(url)
            print("\n--- Article Sentiment Analysis Results ---")
            print(f"Overall Sentiment: {sentiment_results['overall_sentiment']}")
            print(f"Positive Sentiment: {sentiment_results['positive_percentage']:.2f}%")
            print(f"Negative Sentiment: {sentiment_results['negative_percentage']:.2f}%")
        except Exception as e:
            print(f"Error analyzing the article: {e}")

    elif choice == '2':
        query = input("\nEnter the topic to analyze: ")
        print(f"\nAnalyzing information related to '{query}'...")

        try:
            combined_positive, combined_negative = analyze_topic(query, API_KEY, CX)
            print("\n--- Topic Sentiment Analysis Results ---")
            if combined_positive > combined_negative:
                print("Overall Sentiment: POSITIVE")
            elif combined_negative > combined_positive:
                print("Overall Sentiment: NEGATIVE")
            else:
                print("Overall Sentiment: NEUTRAL")
            print(f"Overall Positive Sentiment: {combined_positive:.2f}%")
            print(f"Overall Negative Sentiment: {combined_negative:.2f}%")
        except Exception as e:
            print(f"Error analyzing the topic: {e}")

    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()