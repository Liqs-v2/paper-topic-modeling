from bertopic import BERTopic
import openai
from bertopic.representation import OpenAI
import os
import requests
from typing import Any, Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import re
import csv
import ast
from pathlib import Path
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parse_papers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PARSED_PAPERS_FILE = 'parsed_accepted_papers.csv'
DEFAULT_RAW_PAPERS_FILE = 'accepted_papers.txt'
MAX_WORKERS = 4
ARXIV_API_BASE_URL = 'http://export.arxiv.org/api/query'
REQUEST_TIMEOUT = 10

class Paper:
    def __init__(self, title: str, authors: List[str], abstract: Optional[str] = None):
        self.title = title
        self.authors = authors
        self.abstract = abstract

    def get_escaped_title(self) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '+', self.title)

    def __eq__(self, other):
        if not isinstance(other, Paper):
            return False
        return (self.title == other.title and self.authors == other.authors)

    def __str__(self):
        return f"{self.title}\n{self.abstract}"

    def __repr__(self):
        return f"Paper(title='{self.title}', abstract='{self.abstract}', authors={self.authors})"

def fetch_abstract_for_paper(paper: Paper) -> None:
    """Fetch and set the abstract for a paper from arXiv."""
    metadata = _fetch_arxiv_metadata_for(paper)
    title, abstract = _parse_arxiv_response(metadata)

    if not abstract and not title:
        logger.warning(f'Did not get any results on arXiv for {paper.title}, using empty abstract.')
        paper.abstract = ''
        return

    if title != paper.title:
        logger.warning(f'Could not find {paper.title} on arXiv, using empty abstract.')
        paper.abstract = ''
        return

    paper.abstract = abstract
    return

def _fetch_arxiv_metadata_for(paper: Paper) -> str:
    """Fetch metadata from arXiv API for a given paper."""
    url = f'{ARXIV_API_BASE_URL}?search_query=ti:{paper.get_escaped_title()}&start=0&max_results=1'
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching metadata for {paper.title}: {e}")
        return ""

def _parse_arxiv_response(metadata: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse arXiv API response to extract title and abstract."""
    title = None
    abstract = None

    soup = BeautifulSoup(markup=metadata, features='xml')
    entry_tag = soup.find('entry')

    if not entry_tag:
        return title, abstract

    abstract_tag = entry_tag.find('summary')
    if abstract_tag:
        abstract = re.sub(r'\s+', ' ', abstract_tag.get_text().strip())

    title_tag = entry_tag.find('title')
    if title_tag:
        title = re.sub(r'\s+', ' ', title_tag.get_text().strip())
    return title, abstract

def load_parsed_papers(path_to_parsed_papers: Path) -> List[Paper]:
    """Load previously parsed papers from CSV file."""
    papers = []
    if not path_to_parsed_papers.exists():
        logger.info(f"Parsed papers file not found at {path_to_parsed_papers}. Starting fresh.")
        return []

    try:
        with open(path_to_parsed_papers, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)

            for row in csv_reader:
                try:
                    papers.append(Paper(row['title'], ast.literal_eval(row['authors']), row['abstract']))
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse authors for paper '{row['title']}': {e}. Skipping.")
                    continue

        logger.info(f'Loaded {len(papers)} parsed papers.')
    except Exception as e:
        logger.error(f"Error loading parsed papers from {path_to_parsed_papers}: {e}")
        return []

    return papers

def load_raw_papers(existing_papers: List[Paper], path_to_raw_papers: Path) -> Tuple[List[Paper], bool]:
    """Load raw papers from HTML file and merge with existing papers."""
    if not path_to_raw_papers.exists():
        logger.warning('Raw papers file not found. Cannot load additional papers.')
        return existing_papers

    papers = existing_papers.copy()
    
    try:
        with open(path_to_raw_papers, "r", encoding='utf-8') as f:
            data = f.read()

        soup = BeautifulSoup(markup=data, features='html.parser')
        new_papers_count = 0
        
        for li in soup.find_all('li'):
            title_tag = li.find('strong')
            if title_tag:
                title = title_tag.get_text().strip()

                authors_tag = li.find('em')
                if authors_tag:
                    authors_text = authors_tag.get_text().strip()
                    authors = [author.strip() for author in authors_text.split(',')]

                    paper = Paper(title, authors)

                    if paper not in papers:
                        papers.append(paper)
                        new_papers_count += 1
        
        logger.info(f'Loaded {new_papers_count} new papers from raw file.')

    except Exception as e:
        logger.error(f"Error loading raw papers from {path_to_raw_papers}: {e}")

    return papers, new_papers_count > 0

def fetch_papers(path_to_parsed_papers: Path, path_to_raw_papers: Path) -> Tuple[List[Paper], bool]:
    """Fetch papers from both parsed and raw sources."""
    papers = load_parsed_papers(path_to_parsed_papers)
    return load_raw_papers(papers, path_to_raw_papers)

def process_papers_with_abstracts(papers: List[Paper], output_file: Path) -> None:
    """Process papers to fetch abstracts and save to CSV file."""
    if not papers:
        logger.warning('No papers to process.')
        return

    logger.info(f'Processing {len(papers)} papers to fetch abstracts.')
    
    # Get field names from the first paper
    fields = list(vars(papers[0]).keys())
    lock = threading.Lock()

    def process_single_paper(paper: Paper) -> Dict[str, Any]:
        """Process a single paper to ensure it has an abstract."""
        if not paper.abstract:
            fetch_abstract_for_paper(paper)
        return vars(paper)

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for doc_dict in tqdm(
                    executor.map(process_single_paper, papers), 
                    total=len(papers), 
                    desc='Fetching abstracts from arXiv'
                ):
                    with lock:
                        writer.writerow(doc_dict)
                        
        logger.info(f'Successfully processed and saved {len(papers)} papers to {output_file}.')
        
    except Exception as e:
        logger.error(f"Error processing papers: {e}")
        raise

def load_papers():
    """Main function that orchestrates the paper processing workflow."""
    logger.info("Starting paper processing workflow...")
    
    # Define file paths
    path_to_parsed_papers = Path(DEFAULT_PARSED_PAPERS_FILE)
    path_to_raw_papers = Path(DEFAULT_RAW_PAPERS_FILE)
    
    try:
        papers, has_new_papers = fetch_papers(path_to_parsed_papers, path_to_raw_papers)
        
        if not papers:
            logger.warning('No papers found. Exiting.')
            return
        
        logger.info(f'Total papers loaded: {len(papers)}')

        if has_new_papers:
            process_papers_with_abstracts(papers, path_to_parsed_papers)

        papers_without_abstracts = [paper for paper in papers if not paper.abstract]
        logger.info(f'Papers without abstracts: {len(papers_without_abstracts)}')

        return papers
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
        raise

if __name__ == "__main__":
    load_papers()