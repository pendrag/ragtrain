#!/usr/bin/env python3
"""
ArXiv High Energy Physics Question Generator

This script:
1. Fetches random high energy physics papers from arXiv
2. Downloads and parses the full PDF content
3. Extracts all paragraphs 
4. Identifies informational categories in each paragraph
5. Generates appropriate questions based on the content
6. Outputs (query, category, paragraph) triplets
"""

import os
import re
import random
import requests
import time
import arxiv
import spacy
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import PyPDF2
from tqdm import tqdm
import pdfplumber
import fitz  # PyMuPDF


# Load NLP model for entity recognition and text processing
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

# Define query categories and their patterns
QUERY_CATEGORIES = {
    "THEORETICAL_CONCEPT": {
        "name": "Theoretical Concept",
        "patterns": [
            r'\btheory\b', r'\btheorem\b', r'\bmodel\b', r'\bframework\b', r'\bprinciple\b', 
            r'\bsymmetry\b', r'\binvariance\b', r'\bgauge\b', r'\bfield\b', r'\bparticle\b',
            r'\bboson\b', r'\bfermion\b', r'\bstandard model\b', r'\bquantum\b', r'\bsupersymmetry\b',
            r'\bstring theory\b', r'\bconformal\b', r'\bholographic\b', r'\brenormalization\b'
        ]
    },
    "EXPERIMENTAL_RESULT": {
        "name": "Experimental Result",
        "patterns": [
            r'\bexperiment\b', r'\bmeasurement\b', r'\bobservation\b', r'\bdetection\b', 
            r'\bdiscover(y|ed|ies)?\b', r'\bevidence\b', r'\bdata\b', r'\bresult\b', r'\bsignal\b',
            r'\bcollision\b', r'\bdecay\b', r'\bcross section\b', r'\bluminosity\b', r'\bdetector\b',
            r'\bstatistical\b', r'\bconfidence level\b', r'\bsignificance\b', r'\bprecision\b'
        ]
    },
    "METHODOLOGY": {
        "name": "Methodology",
        "patterns": [
            r'\bmethod\b', r'\bapproach\b', r'\btechnique\b', r'\bprocedure\b', r'\balgorithm\b', 
            r'\bsimulation\b', r'\bcalculation\b', r'\bcomputation\b', r'\banalysis\b',
            r'\bmonte carlo\b', r'\bnumerical\b', r'\banalytical\b', r'\bperturbation\b', r'\brenormalization\b',
            r'\bintegration\b', r'\bexpansion\b', r'\bapproximation\b', r'\bregularization\b'
        ]
    },
    "IMPLICATION": {
        "name": "Implication or Significance",
        "patterns": [
            r'\bimplication\b', r'\bsignificance\b', r'\bconsequence\b', r'\bmeaning\b', 
            r'\bimportant\b', r'\bsuggest\b', r'\bindicate\b', r'\bpoint to\b',
            r'\bsupport\b', r'\bchallenge\b', r'\bconsistent with\b', r'\binconsistent with\b', r'\bprediction\b',
            r'\btherefore\b', r'\bhence\b', r'\bthus\b', r'\bconclusion\b', r'\bdemonstrate\b'
        ]
    },
    "COMPARISON": {
        "name": "Comparison or Relation",
        "patterns": [
            r'\bcompare\b', r'\brelation\b', r'\bcontrast\b', r'\bversus\b', r'\bunlike\b', 
            r'\bsimilar\b', r'\bdifference\b', r'\bagreement\b', r'\bdisagreement\b',
            r'\bcorrelate\b', r'\bcorrespond\b', r'\bmatch\b', r'\bcontradict\b', r'\bconsistent\b',
            r'\bwhile\b', r'\bin contrast\b', r'\bcompared to\b', r'\bwhereas\b', r'\banalogous\b'
        ]
    },
    "HISTORICAL_CONTEXT": {
        "name": "Historical Context",
        "patterns": [
            r'\bhistory\b', r'\bdevelopment\b', r'\bevolution\b', r'\bprogress\b', 
            r'\bsince\b', r'\bpast\b', r'\bdecade\b', r'\bcentury\b', r'\boriginally\b',
            r'\bfirst\b', r'\binitial\b', r'\bpreviously\b', r'\btraditionally\b', r'\boriginally\b',
            r'\brecently\b', r'\bhave been\b', r'\bover the years\b', r'\bsincerely\b', r'\bbackground\b'
        ]
    },
    "FUTURE_RESEARCH": {
        "name": "Future Research",
        "patterns": [
            r'\bfuture\b', r'\bprospect\b', r'\bnext\b', r'\bupcoming\b', r'\bproposal\b', 
            r'\bsuggest\b', r'\brecommend\b', r'\bdirection\b', r'\bimplication\b',
            r'\bfurther\b', r'\badditional\b', r'\bsubsequent\b', r'\bfollow-up\b', r'\bextension\b',
            r'\bremain\b', r'\bopen question\b', r'\bunanswered\b', r'\bcommence\b', r'\bcontinue\b'
        ]
    },
    "MATHEMATICAL_FORMALISM": {
        "name": "Mathematical Formalism",
        "patterns": [
            r'\bequation\b', r'\bformula\b', r'\bmathematical\b', r'\bexpression\b', r'\bderivation\b',
            r'\bproof\b', r'\bsolution\b', r'\bintegral\b', r'\bderivative\b', r'\bdifferential\b',
            r'\boperator\b', r'\balgebra\b', r'\btensor\b', r'\bmatrix\b', r'\bvector\b',
            r'\bhamilton\w+\b', r'\blagrang\w+\b', r'\bboundary condition\b', r'\bconstraint\b'
        ]
    }
}

# Generate question templates for each category
QUESTION_TEMPLATES = {
    "THEORETICAL_CONCEPT": [
        "How does {concept} contribute to our understanding of {field}?",
        "What are the key principles behind {concept} in high energy physics?",
        "Can you explain the relationship between {concept} and {related_concept}?",
        "Why is {concept} important for {application}?",
        "What evidence supports the validity of {concept}?",
        "How has {concept} evolved in the context of modern physics?",
        "What are the limitations of {concept} in explaining {phenomenon}?",
        "How does {concept} fit within the broader framework of {field}?"
    ],
    "EXPERIMENTAL_RESULT": [
        "What methodology was used to obtain {result}?",
        "How does {result} compare to previous experimental findings?",
        "What are the error margins in the measurement of {result}?",
        "How does {result} confirm or challenge current theoretical models?",
        "What are the implications of {result} for our understanding of {field}?",
        "What experimental setup was used to measure {result}?",
        "How significant is {result} in statistical terms?",
        "What systematic uncertainties affect the measurement of {result}?"
    ],
    "METHODOLOGY": [
        "How does {method} improve upon previous approaches?",
        "What are the limitations of using {method}?",
        "Why is {method} appropriate for studying {phenomenon}?",
        "How can {method} be combined with {other_method} for better results?",
        "What assumptions are made when applying {method}?",
        "How computationally intensive is {method} compared to alternatives?",
        "What validation techniques ensure the reliability of {method}?",
        "How widely adopted is {method} in the field of {field}?"
    ],
    "IMPLICATION": [
        "How does {finding} challenge our current understanding of {field}?",
        "What are the broader implications of {finding} for {related_field}?",
        "How might {finding} influence future research directions?",
        "Does {finding} resolve any long-standing questions in the field?",
        "What practical applications might result from {finding}?",
        "How does {finding} connect to other areas of physics?",
        "What theoretical frameworks are supported or weakened by {finding}?",
        "How does {finding} change our perspective on {concept}?"
    ],
    "COMPARISON": [
        "What are the key differences between {subject_a} and {subject_b}?",
        "How do the predictions of {theory_a} and {theory_b} differ regarding {phenomenon}?",
        "Why does {approach_a} yield different results than {approach_b}?",
        "How compatible are {concept_a} and {concept_b}?",
        "Under what conditions would {model_a} be preferred over {model_b}?",
        "What experimental results distinguish between {theory_a} and {theory_b}?",
        "How have perspectives on {subject_a} versus {subject_b} changed over time?",
        "What are the relative strengths and weaknesses of {approach_a} and {approach_b}?"
    ],
    "HISTORICAL_CONTEXT": [
        "How has our understanding of {concept} evolved since its initial proposal?",
        "What were the key milestones in the development of {field}?",
        "How did {scientist}'s work influence the current understanding of {concept}?",
        "What historical experiments led to the acceptance of {theory}?",
        "How has the focus in {field} shifted over the past decades?",
        "What theoretical challenges has {concept} faced throughout its history?",
        "How has technological advancement influenced our ability to study {phenomenon}?",
        "What scientific controversies surrounded the early development of {theory}?"
    ],
    "FUTURE_RESEARCH": [
        "What are the most promising directions for future research on {topic}?",
        "What technological advancements would enable better investigation of {phenomenon}?",
        "What unresolved questions remain about {topic}?",
        "How might future experiments test the predictions of {theory}?",
        "What interdisciplinary approaches might yield new insights into {topic}?",
        "What computational advances would accelerate progress in understanding {phenomenon}?",
        "What are the major obstacles to further progress in {field}?",
        "How might upcoming facilities or instruments advance our understanding of {topic}?"
    ],
    "MATHEMATICAL_FORMALISM": [
        "How is {equation} derived from first principles?",
        "What physical significance does each term in {equation} represent?",
        "How does {formalism} help us understand {phenomenon}?",
        "What approximations are made in deriving {equation}?",
        "How does {equation} behave in extreme or limiting cases?",
        "What are the boundary conditions for applying {formalism}?",
        "How can {equation} be numerically solved in practical applications?",
        "What symmetries are manifest or hidden in {formalism}?"
    ]
}

# Physics-specific terminology for entity extraction
PHYSICS_TERMS = [
    # Particles
    "quark", "lepton", "boson", "fermion", "hadron", "meson", "baryon",
    "gluon", "photon", "electron", "neutrino", "proton", "neutron", "muon", "tau",
    "W boson", "Z boson", "Higgs boson", "graviton", "axion", "tachyon",
    
    # Fundamental concepts
    "symmetry", "conservation", "field", "quantum", "relativity", "gravity",
    "electromagnetism", "strong force", "weak force", "standard model",
    "supersymmetry", "string theory", "M-theory", "quantum field theory",
    "gauge theory", "Higgs", "dark matter", "dark energy", "cosmic",
    
    # Experimental terms
    "particle", "accelerator", "detector", "collision", "decay", "scattering",
    "cross section", "luminosity", "energy", "momentum", "spin", "charge",
    "LHC", "CERN", "Fermilab", "SLAC", "DESY", "RHIC", "Belle", "BaBar",
    
    # Mathematical concepts
    "Lagrangian", "Hamiltonian", "action", "potential", "tensor", "vector",
    "scalar", "spinor", "group", "algebra", "symmetry", "renormalization",
    "perturbation", "non-perturbative", "lattice", "Monte Carlo", "path integral",
    
    # Theoretical frameworks
    "QCD", "QED", "electroweak", "GUT", "TOE", "AdS/CFT", "holography",
    "supergravity", "effective field theory", "operator product expansion",
    "conformal field theory", "quantum gravity", "loop quantum gravity",
    "causal dynamical triangulation", "twistor theory", "bootstrap",
    
    # Cosmology
    "inflation", "big bang", "cosmological constant", "universe", "multiverse",
    "cosmic microwave background", "CMB", "baryon acoustic oscillation", "BAO"
]


class PaperTextExtractor:
    """Extract full text and paragraphs from academic papers in PDF format."""
    
    def __init__(self, verbose=True):
        """Initialize the extractor.
        
        Args:
            verbose: Whether to print detailed progress information
        """
        self.verbose = verbose
        
    def _log(self, message):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
            
    def extract_text_with_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber with column detection.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            # Dictionary to hold text by page number
            all_text = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    self._log(f"Processing page {page_num+1}/{len(pdf.pages)}")
                    
                    # Extract text with more preserving settings
                    text = page.extract_text(
                        x_tolerance=3,  # Horizontal tolerance for character grouping
                        y_tolerance=3,  # Vertical tolerance for line grouping
                    )
                    
                    if text:
                        all_text.append(text)
                    
            return "\n\n".join(all_text)
            
        except Exception as e:
            self._log(f"Error with pdfplumber: {e}")
            return None
            
    def extract_text_with_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF (fitz) which handles academic papers well.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            # Open the PDF file
            doc = fitz.open(pdf_path)
            
            full_text = []
            
            # Iterate through each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text with preserving paragraphs
                text = page.get_text("text")
                
                if text:
                    full_text.append(text)
                    
            # Close the document
            doc.close()
            
            return "\n\n".join(full_text)
            
        except Exception as e:
            self._log(f"Error with PyMuPDF: {e}")
            return None
            
    def extract_text_best_effort(self, pdf_path):
        """Try multiple extraction methods and use the best result.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        # Try PyMuPDF first (usually better for academic papers)
        text = self.extract_text_with_pymupdf(pdf_path)
        
        # If that fails or produces little text, try pdfplumber
        if not text or len(text) < 500:
            self._log("PyMuPDF extraction yielded limited results, trying pdfplumber...")
            text_plumber = self.extract_text_with_pdfplumber(pdf_path)
            
            # Use whichever extraction yielded more text
            if text_plumber and (not text or len(text_plumber) > len(text)):
                text = text_plumber
                
        if not text:
            self._log("All text extraction methods failed.")
        else:
            self._log(f"Successfully extracted {len(text)} characters.")
            
        return text or ""
        
    def clean_text(self, text):
        """Clean extracted text for better paragraph identification.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove reference numbers like [1], [2,3], etc.
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers and footers (common in academic papers)
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            # Skip short header/footer lines
            if len(line.strip()) < 60 and (i == 0 or i == len(lines) - 1):
                continue
                
            # Skip lines that are likely page headers/footers
            if re.match(r'^[0-9]+$', line.strip()):  # Just a page number
                continue
                
            cleaned_lines.append(line)
            
        text = '\n'.join(cleaned_lines)
        
        return text
        
    def identify_paragraphs(self, text):
        """Identify distinct paragraphs from cleaned text.
        
        This is more robust for academic papers with sections, equations, etc.
        
        Args:
            text: Cleaned text from PDF
            
        Returns:
            List of paragraphs
        """
        # Ensure we're working with clean text
        text = self.clean_text(text)
        
        # First split by double newlines (most reliable paragraph separator)
        chunks = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # Further processing to identify and clean paragraphs
        paragraphs = []
        for chunk in chunks:
            # Skip if it's too short to be a meaningful paragraph
            if len(chunk) < 100:
                continue
                
            # Skip if it looks like a header, figure caption, table, etc.
            if re.match(r'^(Fig\.|Figure|Table|References|Bibliography|Appendix|Chapter|Section)\b', chunk):
                continue
                
            # Skip if it has too many special characters or numbers (likely equations)
            alpha_ratio = sum(c.isalpha() for c in chunk) / max(1, len(chunk))
            if alpha_ratio < 0.5:
                continue
                
            # Check if the text has sentence structure (contains periods followed by spaces)
            if not re.search(r'\.\s+[A-Z]', chunk) and not re.search(r'\.\s*$', chunk):
                continue
                
            # It's likely a proper paragraph at this point
            paragraphs.append(chunk)
            
        return paragraphs
        
    def extract_sections(self, text):
        """Attempt to extract sections with headings.
        
        Args:
            text: Full text from PDF
            
        Returns:
            Dictionary mapping section titles to section content
        """
        # Common section headings in academic papers
        section_patterns = [
            r'\n([0-9]+\.[0-9]+\s+[A-Z][^.]+)\n',  # Numbered subsections like "1.1 Introduction"
            r'\n([0-9]+\.\s+[A-Z][^.]+)\n',         # Numbered sections like "1. Introduction"
            r'\n([A-Z][A-Z\s]+)\n',                 # ALL CAPS headings
            r'\n((?:[A-Z][a-z]*\s*){1,4})\n'        # Title Case Headings (up to 4 words)
        ]
        
        # Find potential section breaks
        sections = {}
        last_end = 0
        last_heading = "Abstract"
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                heading = match.group(1).strip()
                start = match.end()
                
                # If we found a new section, store the previous one
                if start > last_end and last_end > 0:
                    section_content = text[last_end:match.start()].strip()
                    if section_content and len(section_content) > 150:  # Avoid empty/tiny sections
                        sections[last_heading] = section_content
                
                last_heading = heading
                last_end = start
                
        # Add the final section
        if last_end > 0 and last_end < len(text):
            section_content = text[last_end:].strip()
            if section_content and len(section_content) > 150:
                sections[last_heading] = section_content
                
        return sections
        
    def process_paper(self, pdf_path):
        """Process a paper and extract text, paragraphs, and sections.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted content
        """
        self._log(f"Processing paper: {pdf_path}")
        
        # Extract full text
        full_text = self.extract_text_best_effort(pdf_path)
        
        if not full_text:
            self._log("Failed to extract text from the paper.")
            return {
                "success": False,
                "full_text": "",
                "paragraphs": [],
                "sections": {}
            }
            
        # Clean the text
        cleaned_text = self.clean_text(full_text)
        
        # Extract paragraphs
        paragraphs = self.identify_paragraphs(cleaned_text)
        self._log(f"Identified {len(paragraphs)} paragraphs.")
        
        # Try to extract sections
        sections = self.extract_sections(full_text)
        self._log(f"Identified {len(sections)} sections.")
        
        return {
            "success": True,
            "full_text": cleaned_text,
            "paragraphs": paragraphs,
            "sections": sections
        }

class ArXivPhysicsQAGenerator:
    """Generates question-answer pairs from arXiv high energy physics papers."""
    
    def __init__(self, temp_dir=None, max_papers=5, verbose=True):
        """Initialize the generator.
        
        Args:
            temp_dir: Directory to store temporary files
            max_papers: Maximum number of papers to process
            verbose: Whether to print detailed progress information
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.max_papers = max_papers
        self.verbose = verbose
        self.pdf_dir = os.path.join(self.temp_dir, "pdfs")
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for category, data in QUERY_CATEGORIES.items():
            self.compiled_patterns[category] = [re.compile(pattern) for pattern in data["patterns"]]
            
        # Create a set of physics terms for faster lookup
        self.physics_terms_set = set(term.lower() for term in PHYSICS_TERMS)
        
    def _log(self, message):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
            
    def fetch_arxiv_papers(self, categories=None, max_results=None):
        """Fetch papers from arXiv API.
        
        Args:
            categories: List of arXiv categories to search
            max_results: Maximum number of results to return
            
        Returns:
            List of arxiv.Result objects
        """
        if categories is None:
            categories = ["hep-th", "hep-ph", "hep-ex", "hep-lat"]
            
        max_results = max_results or self.max_papers
        
        # Create search query for specified categories
        query = " OR ".join(f"cat:{cat}" for cat in categories)
        
        self._log(f"Searching arXiv for: {query}")
        
        # Use the arxiv.Search class to define the query
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(search.results())
 
        self._log(f"Found {len(results)} papers")
        
        return results
        
    def download_papers(self, papers):
        """Download PDFs for the given papers.
        
        Args:
            papers: List of arxiv.Result objects
            
        Returns:
            Dictionary mapping paper IDs to PDF file paths
        """
        paper_files = {}
        
        for paper in tqdm(papers, desc="Downloading papers", disable=not self.verbose):
            # Extract the paper ID for filename
            paper_id = paper.entry_id.split("/")[-1].replace(".", "_")
            pdf_path = os.path.join(self.pdf_dir, f"{paper_id}.pdf")
            
            # Download the PDF if it doesn't exist
            if not os.path.exists(pdf_path):
                self._log(f"Downloading: {paper.title}")
                try:
                    paper.download_pdf(dirpath=self.pdf_dir, filename=f"{paper_id}.pdf")
                    time.sleep(1)  # Be nice to arXiv servers
                except Exception as e:
                    self._log(f"Failed to download {paper.title}: {e}")
                    continue
            else:
                self._log(f"Using cached PDF for: {paper.title}")
                
            paper_files[paper_id] = {
                "pdf_path": pdf_path,
                "title": paper.title,
                "abstract": paper.summary,
                "authors": [author.name for author in paper.authors],
                "url": paper.entry_id
            }
            
        return paper_files
        
    def extract_paragraphs_from_pdf(self, pdf_path):
        """Extract full text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """

        def replace_newlines(text):
            import re
            # Replace newlines preceded by a hyphen with an empty string
            text = re.sub(r'-\n', '', text)
            # Replace all other newlines with a space
            text = text.replace('\n', ' ')
            return text

        texts = []
        try:
            extractor = PaperTextExtractor(verbose=self.verbose)
            result = extractor.process_paper(pdf_path)
            if result["success"]:
                texts = list(map(replace_newlines, result["sections"].values()))
        except Exception as e:
            self._log(f"Error extracting text from PDF: {e}")
            return ""
        return texts
        
    def extract_key_terms(self, paragraph):
        """Extract key physics terms from a paragraph using NLP.
        
        Args:
            paragraph: Text paragraph to analyze
            
        Returns:
            List of extracted key terms
        """
        # Process the paragraph with spaCy
        doc = nlp(paragraph)
        
        # Extract relevant terms using various approaches
        extracted_terms = set()
        
        # 1. Extract named entities
        for ent in doc.ents:
            if len(ent.text) > 2:  # Filter out very short entities
                extracted_terms.add(ent.text.lower())
        
        # 2. Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2:
                extracted_terms.add(chunk.text.lower())
        
        # 3. Look for known physics terms
        for term in self.physics_terms_set:
            if term in paragraph.lower():
                extracted_terms.add(term)
        
        # Sort by length (longer terms first) and take top terms
        return sorted(list(extracted_terms), key=len, reverse=True)[:5]
        
    def identify_categories(self, paragraph):
        """Identify which information categories apply to a paragraph.
        
        Args:
            paragraph: Text paragraph to analyze
            
        Returns:
            List of category keys that apply to the paragraph
        """
        categories = []
        
        # Check each category's patterns against the paragraph
        for category, patterns in self.compiled_patterns.items():
            # If any pattern matches, add the category
            if any(pattern.search(paragraph.lower()) for pattern in patterns):
                categories.append(category)
                
        return categories
        
    def generate_questions(self, paragraph, categories):
        """Generate questions for a paragraph based on its categories.
        
        Args:
            paragraph: Text paragraph to generate questions for
            categories: List of category keys that apply to the paragraph
            
        Returns:
            List of dictionaries with 'question', 'category', and 'paragraph' keys
        """
        results = []
        key_terms = self.extract_key_terms(paragraph)
        
        # Skip if no key terms found
        if not key_terms:
            return results
            
        # For each applicable category
        for category in categories:
            # Skip if no templates for this category
            if category not in QUESTION_TEMPLATES:
                continue
                
            # Generate up to 3 questions per category
            templates = QUESTION_TEMPLATES[category]
            num_questions = min(3, len(templates))
            
            for i in range(num_questions):
                template = random.choice(templates)
                
                # Create a dictionary of replacements
                replacements = {
                    "{concept}": key_terms[0] if key_terms else "this concept",
                    "{field}": "high energy physics",
                    "{topic}": key_terms[0] if key_terms else "this topic",
                    "{phenomenon}": key_terms[0] if key_terms else "this phenomenon",
                    "{result}": "these results",
                    "{method}": "this method",
                    "{finding}": "this finding",
                    "{equation}": "this equation",
                    "{formalism}": "this formalism"
                }
                
                # Add secondary terms if available
                if len(key_terms) > 1:
                    replacements.update({
                        "{related_concept}": key_terms[1],
                        "{related_field}": key_terms[1],
                        "{application}": key_terms[1],
                        "{subject_a}": key_terms[0],
                        "{subject_b}": key_terms[1],
                        "{theory_a}": key_terms[0],
                        "{theory_b}": key_terms[1],
                        "{approach_a}": key_terms[0],
                        "{approach_b}": key_terms[1],
                        "{model_a}": key_terms[0],
                        "{model_b}": key_terms[1],
                        "{other_method}": key_terms[1]
                    })
                    
                # Add tertiary terms if available
                if len(key_terms) > 2:
                    replacements.update({
                        "{scientist}": key_terms[2]
                    })
                else:
                    replacements["{scientist}"] = "researchers"
                    
                # Replace placeholders with actual terms
                question = template
                for placeholder, replacement in replacements.items():
                    question = question.replace(placeholder, replacement)
                    
                # Add the question to results if it's not a duplicate
                if not any(q['question'] == question for q in results):
                    results.append({
                        "question": question,
                        "category": QUERY_CATEGORIES[category]["name"],
                        "paragraph": paragraph
                    })
                    
        return results
        
    def process_papers(self):
        """Main method to process papers and generate Q&A pairs.
        
        Returns:
            List of dictionaries with 'question', 'category', and 'paragraph' keys
        """
        all_results = []
        
        try:
            # Fetch papers
            self._log("Fetching papers from arXiv...")
            papers = self.fetch_arxiv_papers()
            
            if not papers:
                raise ValueError("No papers found. Try again later.")
                
            # Download papers
            paper_files = self.download_papers(papers)
            
            # Process each paper
            for paper_id, paper_info in paper_files.items():
                self._log(f"\nProcessing paper: {paper_info['title']}")
                
                # Extract text from PDF
                paragraphs = self.extract_paragraphs_from_pdf(paper_info['pdf_path'])
                self._log(f"  Extracted {len(paragraphs)} paragraphs.")
                
                paper_results = []
                
                # Process each paragraph
                for paragraph in tqdm(paragraphs, desc="Processing paragraphs", disable=not self.verbose):
                    # Skip paragraphs that are too short
                    if len(paragraph) < 100:
                        continue
                        
                    # Identify categories
                    categories = self.identify_categories(paragraph)
                    
                    if categories:
                        # Generate questions
                        questions = self.generate_questions(paragraph, categories)
                        paper_results.extend(questions)
                        
                self._log(f"  Generated {len(paper_results)} Q&A pairs for this paper.")
                all_results.extend(paper_results)
                
            self._log(f"\nGenerated a total of {len(all_results)} Q&A pairs.")
            
            # Output sample results
            if self.verbose and all_results:
                self._log("\nSample Q&A Pairs:")
                for i in range(min(5, len(all_results))):
                    self._log(f"\n[{i+1}] Question: {all_results[i]['question']}")
                    self._log(f"    Category: {all_results[i]['category']}")
                    self._log(f"    Paragraph: \"{all_results[i]['paragraph'][:150]}...\"")
                    
            return all_results
            
        except Exception as e:
            self._log(f"Error: {e}")
            return all_results
            
    def save_results(self, results, output_file):
        """Save results to a file.
        
        Args:
            results: List of dictionaries with 'question', 'category', and 'paragraph' keys
            output_file: Path to the output file
        """
        import json
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        self._log(f"Results saved to {output_file}")


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from arXiv high energy physics papers")
    parser.add_argument("--output", "-o", default="physics_qa_results.json", help="Output file path")
    parser.add_argument("--papers", "-p", type=int, default=5, help="Number of papers to process")
    parser.add_argument("--categories", "-c", nargs="+", default=["hep-th", "hep-ph", "hep-ex", "hep-lat"], 
                        help="ArXiv categories to search")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Create and run the generator
    generator = ArXivPhysicsQAGenerator(max_papers=args.papers, verbose=not args.quiet)
    results = generator.process_papers()
    
    # Save results
    generator.save_results(results, args.output)
    
    print(f"Generated {len(results)} Q&A pairs from {args.papers} papers.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()