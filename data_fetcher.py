import requests
from bs4 import BeautifulSoup
    
def search_google(query, api_key, cx):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}"
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