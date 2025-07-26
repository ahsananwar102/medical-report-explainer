"""
Data ingestion module for medical corpus creation
Downloads PubMed abstracts and Mayo Clinic articles
"""

import json
import time
import requests
import ssl
import urllib3
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from Bio import Entrez
import pandas as pd

from config.config import (
    PUBMED_EMAIL, MAX_PUBMED_ARTICLES, MAYO_CLINIC_BASE_URL,
    CORPUS_DIR, DATA_DIR
)

# Disable SSL warnings and set up SSL context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context


class PubMedDownloader:
    """Download and process PubMed abstracts"""
    
    def __init__(self, email: str = PUBMED_EMAIL):
        """Initialize PubMed downloader with email for Entrez API"""
        Entrez.email = email
        self.articles = []
        
    def search_articles(self, query: str, max_results: int = MAX_PUBMED_ARTICLES) -> List[str]:
        """
        Search PubMed for articles matching query
        
        Args:
            query: Search query string
            max_results: Maximum number of articles to retrieve
            
        Returns:
            List of PubMed IDs
        """
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            return search_results["IdList"]
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch abstracts for given PubMed IDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article dictionaries with title, abstract, keywords
        """
        articles = []
        batch_size = 100  # Process in batches to avoid API limits
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch),
                    rettype="abstract",
                    retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()
                
                for record in records["PubmedArticle"]:
                    article = self._parse_article(record)
                    if article:
                        articles.append(article)
                        
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching batch: {e}")
                continue
                
        return articles
    
    def _parse_article(self, record) -> Optional[Dict]:
        """Parse PubMed article record"""
        try:
            article = record["MedlineCitation"]["Article"]
            
            # Extract title
            title = article.get("ArticleTitle", "")
            
            # Extract abstract
            abstract_sections = article.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_sections, list):
                abstract = " ".join([str(section) for section in abstract_sections])
            else:
                abstract = str(abstract_sections)
            
            # Extract keywords
            keywords = []
            keyword_list = record["MedlineCitation"].get("KeywordList", [])
            for kw_group in keyword_list:
                for keyword in kw_group:
                    keywords.append(str(keyword))
            
            # Extract MeSH terms
            mesh_terms = []
            mesh_list = record["MedlineCitation"].get("MeshHeadingList", [])
            for mesh in mesh_list:
                descriptor = mesh.get("DescriptorName", {})
                if hasattr(descriptor, 'attributes') and descriptor.attributes:
                    mesh_terms.append(str(descriptor))
            
            return {
                "pmid": record["MedlineCitation"]["PMID"],
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
                "mesh_terms": mesh_terms,
                "source": "pubmed"
            }
            
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None
    
    def download_medical_corpus(self) -> List[Dict]:
        """Download comprehensive medical corpus from PubMed"""
        medical_queries = [
            "medical terminology",
            "clinical symptoms",
            "disease diagnosis",
            "medical procedures",
            "anatomy physiology",
            "pharmacology drugs",
            "medical imaging",
            "laboratory tests",
            "patient care",
            "medical conditions"
        ]
        
        all_articles = []
        for query in medical_queries:
            print(f"Searching for: {query}")
            try:
                pmids = self.search_articles(query, max_results=100)
                if pmids:
                    articles = self.fetch_abstracts(pmids)
                    all_articles.extend(articles)
                    print(f"Downloaded {len(articles)} articles for '{query}'")
                else:
                    print(f"No articles found for '{query}'")
            except Exception as e:
                print(f"Failed to download articles for '{query}': {e}")
                continue
            
        # Remove duplicates based on PMID
        unique_articles = {article["pmid"]: article for article in all_articles}
        return list(unique_articles.values())


class MayoClinicScraper:
    """Scrape Mayo Clinic articles for medical information"""
    
    def __init__(self, base_url: str = MAYO_CLINIC_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        # Disable SSL verification for Mayo Clinic if needed
        self.session.verify = False
    
    def get_disease_articles(self) -> List[Dict]:
        """
        Scrape Mayo Clinic disease and condition articles
        Note: This is a simplified implementation for demonstration
        """
        # Common medical conditions - URLs from Mayo Clinic
        condition_urls = [
            "/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
            "/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410",
            "/diseases-conditions/heart-disease/symptoms-causes/syc-20353118",
            "/diseases-conditions/asthma/symptoms-causes/syc-20369653",
            "/diseases-conditions/arthritis/symptoms-causes/syc-20350772"
        ]
        
        articles = []
        for url in condition_urls:
            try:
                full_url = urljoin(self.base_url, url)
                response = self.session.get(full_url, timeout=10)
                
                if response.status_code == 200:
                    article = self._parse_mayo_page(response.text, full_url)
                    if article:
                        articles.append(article)
                
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
                
        return articles
    
    def _parse_mayo_page(self, html: str, url: str) -> Optional[Dict]:
        """Parse Mayo Clinic page content"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Extract main content
            content_sections = soup.find_all(['p', 'li'], class_=lambda x: x != 'navigation')
            content = " ".join([p.get_text().strip() for p in content_sections if p.get_text().strip()])
            
            return {
                "title": title,
                "content": content,
                "url": url,
                "source": "mayo_clinic"
            }
            
        except Exception as e:
            print(f"Error parsing Mayo Clinic page: {e}")
            return None


def create_sample_medical_corpus():
    """Create a sample medical corpus with predefined data for demo purposes"""
    sample_articles = [
        {
            "pmid": "sample_001",
            "title": "Understanding Hypertension: Causes, Symptoms, and Treatment",
            "abstract": "Hypertension, commonly known as high blood pressure, is a chronic medical condition in which the blood pressure in the arteries is persistently elevated. It is often called the 'silent killer' because it typically has no symptoms but can lead to serious health complications including heart disease, stroke, and kidney problems. Primary hypertension has no identifiable cause and develops gradually over years. Secondary hypertension is caused by underlying conditions such as kidney disease, adrenal gland disorders, or certain medications. Treatment typically involves lifestyle modifications including dietary changes, regular exercise, weight management, and stress reduction. Medications may be prescribed when lifestyle changes are insufficient.",
            "keywords": ["hypertension", "blood pressure", "cardiovascular disease", "treatment"],
            "mesh_terms": ["Hypertension", "Blood Pressure", "Cardiovascular Diseases"],
            "source": "pubmed"
        },
        {
            "pmid": "sample_002", 
            "title": "Diabetes Mellitus: Pathophysiology and Management",
            "abstract": "Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period. Type 1 diabetes results from the pancreas not producing enough insulin. Type 2 diabetes begins with insulin resistance, a condition in which cells fail to respond to insulin properly. The most common symptoms include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many complications including diabetic ketoacidosis, hyperosmolar hyperglycemic state, cardiovascular disease, stroke, chronic kidney disease, foot ulcers, and damage to the eyes. Management includes maintaining blood glucose levels through diet, exercise, and medication when necessary.",
            "keywords": ["diabetes", "insulin", "blood glucose", "metabolism"],
            "mesh_terms": ["Diabetes Mellitus", "Insulin", "Blood Glucose"],
            "source": "pubmed"
        },
        {
            "pmid": "sample_003",
            "title": "Myocardial Infarction: Recognition and Emergency Management",
            "abstract": "Myocardial infarction, commonly known as a heart attack, occurs when blood flow decreases or stops to a part of the heart, causing damage to the heart muscle. The most common symptom is chest pain or discomfort which may travel into the shoulder, arm, back, neck, or jaw. Often it occurs in the center or left side of the chest and lasts for more than a few minutes. The discomfort may occasionally feel like heartburn. Other symptoms may include shortness of breath, nausea, feeling faint, cold sweat, or feeling tired. About 30% of people have atypical symptoms. Women more often present without chest pain and instead have neck pain, arm pain, or feel tired. Risk factors include high blood pressure, smoking, diabetes, lack of exercise, obesity, high blood cholesterol, poor diet, and excessive alcohol intake.",
            "keywords": ["myocardial infarction", "heart attack", "chest pain", "emergency"],
            "mesh_terms": ["Myocardial Infarction", "Acute Coronary Syndrome", "Chest Pain"],
            "source": "pubmed"
        }
    ]
    
    mayo_articles = [
        {
            "title": "Heart Disease Overview",
            "content": "Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects). The term 'heart disease' is often used interchangeably with the term 'cardiovascular disease.' Cardiovascular disease generally refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack, chest pain (angina) or stroke. Other heart conditions, such as those that affect your heart's muscle, valves or rhythm, also are considered forms of heart disease.",
            "url": "https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118",
            "source": "mayo_clinic"
        },
        {
            "title": "Understanding Blood Pressure",
            "content": "Blood pressure is the force of your blood pushing against the walls of your arteries. Each time your heart beats, it pumps blood into the arteries. Your blood pressure is highest when your heart beats, pumping the blood. This is called systolic pressure. When your heart is at rest, between beats, your blood pressure falls. This is called diastolic pressure. Your blood pressure reading uses these two numbers. Usually the systolic number comes before or above the diastolic number. A reading of 119/79 or lower is normal blood pressure. 140/90 or higher is high blood pressure. Between 120 and 139 for the top number, or between 80 and 89 for the bottom number is called prehypertension.",
            "url": "https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410",
            "source": "mayo_clinic"
        }
    ]
    
    return sample_articles, mayo_articles


def create_medical_corpus():
    """Create comprehensive medical corpus by combining PubMed and Mayo Clinic data"""
    print("Starting medical corpus creation...")
    
    # Try to download real data first
    pubmed_articles = []
    mayo_articles = []
    
    try:
        # Download PubMed articles
        print("Downloading PubMed articles...")
        pubmed_downloader = PubMedDownloader()
        pubmed_articles = pubmed_downloader.download_medical_corpus()
        
        # Download Mayo Clinic articles
        print("Downloading Mayo Clinic articles...")
        mayo_scraper = MayoClinicScraper()
        mayo_articles = mayo_scraper.get_disease_articles()
        
    except Exception as e:
        print(f"Error downloading real data: {e}")
    
    # If we couldn't download enough real data, supplement with sample data
    if len(pubmed_articles) < 10:
        print("Using sample medical data to supplement corpus...")
        sample_pubmed, sample_mayo = create_sample_medical_corpus()
        pubmed_articles.extend(sample_pubmed)
        mayo_articles.extend(sample_mayo)
    
    # Combine and save corpus
    corpus = {
        "pubmed_articles": pubmed_articles,
        "mayo_articles": mayo_articles,
        "total_articles": len(pubmed_articles) + len(mayo_articles),
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    # Save to file
    corpus_file = CORPUS_DIR / "medical_corpus.json"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    
    print(f"Medical corpus created with {corpus['total_articles']} articles")
    print(f"Saved to: {corpus_file}")
    
    return corpus


if __name__ == "__main__":
    create_medical_corpus() 