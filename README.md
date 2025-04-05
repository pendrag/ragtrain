# RAG Train

Tools for training a RAG system

## data_synthesis

### Usage

The script can be run from the command line:

```bash 
Copiarpython data_synthesis.py --papers 5 --categories hep-th hep-ph --output results.json
```

### Dependencies

The code requires:

* arxiv (for API access)
* PyPDF2 (for PDF parsing)
* spaCy with 'en_core_web_md' model (for NLP)
* requests (for downloading)
* tqdm (for progress visualization)

Just do:

```bash 
pip install -r requirements.txt
```

### Key Functionality

* Fetching Papers: Retrieves papers from specific arXiv categories
* PDF Processing: Downloads and extracts full text from PDFs
* Paragraph Extraction: Identifies and cleans proper paragraphs
