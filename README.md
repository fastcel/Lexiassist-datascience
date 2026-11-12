# LexiAssist Data Science Phase 1

### Australian Legal Corpus: Data Acquisition, Cleaning, EDA and Preprocessing

LexiAssist is an AI-powered legal research assistant designed to support research using Australian legal documents.

The broader project explores **Retrieval-Augmented Generation (RAG)** to retrieve relevant legal information and produce responses supported by source documents.

This repository contains **Phase 1 of the Data Science project**, focusing on preparing and understanding the Australian legal corpus before developing the retrieval pipeline.

## 1. Project Objectives

The objectives of Phase 1 are to:

- Acquire a reproducible sample of Australian legal documents.
- Investigate dataset structure and data quality.
- Clean and standardize legal text and metadata.
- Perform exploratory data analysis with visualizations.
- Identify corpus imbalances, missing values and document-length patterns.
- Prepare retrieval-ready text and metadata.
- Export validated datasets for subsequent project phases.

## 2. Dataset

### Problem Statement

Australian legal documents can be lengthy and difficult to navigate. LexiAssist aims to support legal research by retrieving relevant information from Australian legal sources using a Retrieval-Augmented Generation approach.

Phase 1 focuses on preparing a clean, structured, and well-understood legal corpus for subsequent retrieval experiments.

### Dataset Features and Target Variable

The original dataset contains 10 attributes:

`version_id`, `type`, `jurisdiction`, `source`, `mime`, `date`, `citation`, `url`, `when_scraped`, and `text`.

After cleaning and preprocessing, the final dataset contains 16 attributes, including document-length statistics, missing-date indicators, publication year, and retrieval-ready text.

**Target variable:** Not applicable in Phase 1. This project focuses on information retrieval rather than supervised classification or regression. No labeled prediction target is defined at this stage.

**Normalization and scaling:** Numerical feature scaling is not required for the current text-preparation workflow. Embedding normalization and other retrieval-specific processing will be considered during Phase 2.

## 3. Repository Structure

```text
Lexiassist-datascience/
    notebooks/
        01_data_acquisition_and_initial_audit.ipynb
        02_data_cleaning.ipynb
        03_exploratory_data_analysis.ipynb
        04_text_preprocessing.ipynb

    data/
        australian_legal_corpus_readable.xlsx

    README.md
    requirements.txt
```

The `notebooks/` directory contains all four Jupyter notebooks used throughout Phase 1.

The `data/` directory contains an Excel-friendly preview of the processed dataset.

The three complete CSV datasets are available as downloadable assets in the repository's [GitHub Release](https://github.com/fastcel/Lexiassist-datascience/releases/tag/v1.0-phase1) due to their larger file sizes.

## 4. Project Workflow

### Notebook 01: Data Acquisition and Initial Audit

[View Notebook 01](notebooks/01_data_acquisition_and_initial_audit.ipynb)

This notebook:

- Loads the corpus from Hugging Face.
- Uses a fixed random seed for reproducible sampling.
- Selects 1,000 legal documents.
- Converts the sample into a Pandas DataFrame.
- Examines initial attributes and document distributions.
- Saves the sampled dataset and acquisition metadata.

### Notebook 02: Data Cleaning

[View Notebook 02](notebooks/02_data_cleaning.ipynb)

This notebook:

- Investigates missing metadata.
- Checks duplicate records, IDs, URLs and text.
- Standardizes metadata and timestamps.
- Normalizes legal text using conservative cleaning techniques.
- Calculates word counts and character counts.
- Identifies document-length outliers without automatically removing them.
- Exports the cleaned corpus.

Legal terminology, citations and document structure are preserved wherever possible.

### Notebook 03: Exploratory Data Analysis

[View Notebook 03](notebooks/03_exploratory_data_analysis.ipynb)

Exploratory Data Analysis examines:

- Legal document type distribution
- Jurisdiction distribution
- Data source distribution
- Publication year patterns
- Document length distribution
- Missing metadata by source
- Relationships between jurisdictions and document types

The notebook includes visualizations, statistical summaries and interpretations relevant to LexiAssist.

### Notebook 04: Text Preprocessing

[View Notebook 04](notebooks/04_text_preprocessing.ipynb)

This notebook prepares the cleaned corpus for future retrieval experiments.

Key tasks include:

- Validating document integrity.
- Preserving original legal text.
- Creating a separate `retrieval_text` attribute by combining citations and document content.
- Preparing publication-year metadata.
- Preserving missing-date indicators.
- Validating and exporting the final preprocessed dataset.
- Creating a formatted Excel preview for easier inspection.

No chunking, embedding generation or retrieval model training is performed in this phase.

## 5. Key EDA Findings

| Metric | Finding |
|---|---|
| Total documents | 1,000 |
| Court decisions | 616 (61.6%) |
| Primary legislation | 169 (16.9%) |
| Secondary legislation | 180 (18.0%) |
| Bills | 35 (3.5%) |
| NSW and Commonwealth documents | 700 (70.0%) |
| Missing publication dates | 107 (10.7%) |
| Publication year range | 1929 to 2024 |
| Median document length | 3,526.5 words |
| Maximum document length | 91,018 words |

### Important Observations

**Document imbalance:** Court decisions represent the majority of the sampled corpus.

**Jurisdiction concentration:** New South Wales and Commonwealth documents account for 70% of the sample.

**Source concentration:** NSW Caselaw and the Federal Court of Australia together contribute 56.1% of the documents.

**Missing metadata:** Missing publication dates are concentrated in particular sources, especially the High Court of Australia.

**Document-length variation:** Some legal documents are substantially longer than others, highlighting the need for careful chunking in future retrieval experiments.

These findings describe the sampled dataset rather than the entire Australian legal system.

## 6. Data Files and Downloads

The project produces four datasets for different stages of analysis and preprocessing.

| File | Description |
|---|---|
| `australian_legal_corpus_sample.csv` | Original 1,000-document sample |
| `australian_legal_corpus_cleaned.csv` | Cleaned legal text and metadata |
| `australian_legal_corpus_preprocessed.csv` | Retrieval-ready text and metadata |
| `australian_legal_corpus_readable.xlsx` | Formatted preview for easy inspection |

### Download Full CSV Datasets

The three complete CSV datasets are hosted as assets in the GitHub Release:

**[LexiAssist Phase 1 Datasets (v1.0-phase1)](https://github.com/fastcel/Lexiassist-datascience/releases/tag/v1.0-phase1)**

Individual downloads:

1. [Download Original Sample CSV](https://github.com/fastcel/Lexiassist-datascience/releases/download/v1.0-phase1/australian_legal_corpus_sample.csv)
2. [Download Cleaned Corpus CSV](https://github.com/fastcel/Lexiassist-datascience/releases/download/v1.0-phase1/australian_legal_corpus_cleaned.csv)
3. [Download Preprocessed Corpus CSV](https://github.com/fastcel/Lexiassist-datascience/releases/download/v1.0-phase1/australian_legal_corpus_preprocessed.csv)

All three CSV files preserve the complete legal text.

The files are provided through GitHub Releases because their sizes exceed the standard GitHub browser upload limit.

### Excel Preview

A formatted Excel version is also available:

[Download Readable Excel Preview](data/australian_legal_corpus_readable.xlsx)

The Excel workbook is intended for human inspection and contains shortened legal-text previews for easier reading.

**Important:** The CSV files contain the complete legal text and should be used for further data processing and retrieval experiments. The Excel preview is not intended to replace the full datasets.

## 7. Technologies and Libraries

The project uses the following technologies:

- Python
- Google Colab and Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Hugging Face Datasets
- OpenPyXL
- Git and GitHub

Required Python dependencies are listed in `requirements.txt`.

## 8. Running the Project

### Environment Setup

Clone or download the repository:

```bash
git clone https://github.com/fastcel/Lexiassist-datascience.git
```

Navigate to the project directory:

```bash
cd Lexiassist-datascience
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Executing the Notebooks

Open the notebooks in Google Colab and execute them in the following order:

1. `01_data_acquisition_and_initial_audit.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_exploratory_data_analysis.ipynb`
4. `04_text_preprocessing.ipynb`

Notebook 01 acquires and samples the source dataset. Subsequent notebooks use the CSV files exported by previous stages.

Alternatively, the complete datasets can be downloaded directly from the [GitHub Release](https://github.com/fastcel/Lexiassist-datascience/releases/tag/v1.0-phase1) without rerunning data acquisition.

The notebooks use Google Colab file-upload and download utilities. Some adjustments may be required when running outside Colab.

## 9. Phase 1 Outcomes

Phase 1 successfully establishes a reproducible workflow for acquiring, cleaning, analyzing and preparing Australian legal documents.

The main outcomes include:

- Acquisition of a 1,000-document sample.
- Identification and investigation of data-quality issues.
- Standardization of text and metadata.
- Exploratory visualizations and statistical analysis.
- Identification of corpus imbalances and coverage limitations.
- Preparation of retrieval-ready text.
- Preservation of original legal content.
- Export of the final validated dataset containing 1,000 documents and 16 attributes.

The resulting dataset provides a foundation for future retrieval experiments.

The analysis also identifies data-quality considerations that will inform subsequent development.

## 10. Future Work: Phase 2

Planned work includes:

- Context-preserving legal document chunking.
- Sparse retrieval using BM25.
- Dense retrieval using embeddings and FAISS.
- Hybrid retrieval and rank fusion.
- Retrieval-Augmented Generation.
- Evaluation of retrieved legal information and citation quality.

These components are outside the scope of the current Phase 1 implementation.

## 11. Disclaimer

LexiAssist is an educational research project. It is not a substitute for professional legal advice.

The sampled corpus is not guaranteed to contain current or comprehensive Australian law. Legal information should be verified using authoritative sources.

**Project:** LexiAssist

**Component:** Data Science Phase 1

**Status:** Data acquisition, cleaning, EDA and preprocessing completed.
