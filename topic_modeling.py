from bertopic import BERTopic
from bertopic.representation import OpenAI
from typing import List
import logging

from parse_papers import load_papers, Paper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('topic_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_topic_model() -> BERTopic:
    """Create and return a BERTopic model instance."""
    return BERTopic()

def run_topic_modeling(papers: List[Paper]) -> None:
    """Run topic modeling on the processed papers."""
    # TODO: Implement topic modeling functionality
    logger.info("Topic modeling functionality not yet implemented.")
    
    # Placeholder for future implementation:
    # topic_model = create_topic_model()
    # documents = [str(paper) for paper in papers]
    # topics, probs = topic_model.fit_transform(documents)

def remove_stopwords(text: str) -> str:
    stopwords = set(ENGLISH_STOP_WORDS)

    # Tokenize by splitting on non-word characters, filter stopwords, and rejoin
    tokens = re.findall(r'\b\w+\b', text)
    filtered = [token for token in tokens if token.lower() not in stopwords]
    return ' '.join(filtered)

def main():
    papers = load_papers()
    topic_model = create_topic_model()

    documents = [remove_stopwords(str(paper)) for paper in papers]

    topics, probs = topic_model.fit_transform(documents)
    topic_labels = topic_model.generate_topic_labels(nr_words=3, separator=", ")
    document_info = topic_model.get_document_info(documents)
    document_info.to_csv('document_info.csv')

if __name__ == "__main__":
    main()