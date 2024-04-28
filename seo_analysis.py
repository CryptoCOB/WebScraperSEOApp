from collections import Counter
import logging
from imports import load_ai_components
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import numpy as np
import cv2
from textblob import TextBlob
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urljoin
import json
import utils
# Load AI components (tokenizer and model are assumed to be configured in the imports file)
tokenizer, model = load_ai_components()

async def download_page(url):
    """Asynchronously fetches a webpage and returns its content."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=30) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error occurred when fetching {url}: {str(e)}")
            return None


async def process_image(image_url, session):
    """Download and process images asynchronously."""
    try:
        async with session.get(image_url) as response:
            image_data = await response.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img_np
    except Exception as e:
        print(f"Failed to process image {image_url}: {e}")
        return None

async def analyze_text(text):
    """Use a pretrained language model for summarization or enhancement."""
    inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=120)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

def analyze_text_content(text):
    inputs = utils.safe_tokenized(tokenizer, text, 2048)
    summary = model.generate(**inputs)
    return tokenizer.decode(summary[0], skip_special_tokens=True)
def analyze_sentiment(text):
    """Analyzes the sentiment of a text using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

async def comprehensive_seo_analysis(url, html_content):
    """Perform a full SEO analysis incorporating advanced techniques."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    enriched_text = await analyze_text(text)
    sentiment_polarity, sentiment_subjectivity = analyze_sentiment(text)
    analysis_results = {
        'url': url,
        'enriched_text': enriched_text,
        'sentiment': {
            'polarity': sentiment_polarity,
            'subjectivity': sentiment_subjectivity
        }
    }
    return analysis_results

def perform_complete_analysis(url, html_content):
    """Performs a comprehensive SEO analysis of a webpage."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    keyword_density = {word: tfidf_matrix[0, idx] for word, idx in zip(feature_names, np.argmax(tfidf_matrix.toarray(), axis=1))}

    enriched_text = asyncio.run(analyze_text(text))
    sentiment_polarity, sentiment_subjectivity = analyze_sentiment(text)
    nlp = spacy.load('en_core_web_sm', disable=['parser'])
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]

    analysis_results = {
        'url': url,
        'title': soup.find('title').text.strip() if soup.find('title') else "No title found",
        'meta_description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else "No description found",
        'keyword_density': keyword_density,
        'llm_summary': enriched_text,
        'sentiment': {
            'polarity': sentiment_polarity,
            'subjectivity': sentiment_subjectivity
        },
        'named_entities': named_entities
    }

    return analysis_results

async def main():
    url = "https://example.com"
    html_content = await download_page(url)
    if html_content:
        comprehensive_report = await comprehensive_seo_analysis(url, html_content)
        complete_report = perform_complete_analysis(url, html_content)
        print("Comprehensive SEO Analysis Report:")
        print(json.dumps(comprehensive_report, indent=4))
        print("\nComplete SEO Analysis Report:")
        print(json.dumps(complete_report, indent=4))

if __name__ == "__main__":
    asyncio.run(main())
