"# Text-Mining

A comprehensive collection of Natural Language Processing (NLP) and Text Mining workshops demonstrating various techniques for text analysis, translation assistance, plagiarism detection, and document classification.

## 📖 Description

This repository contains three practical workshops (Ateliers) that explore different aspects of text mining and natural language processing. Each workshop is implemented as an interactive Jupyter notebook, providing hands-on experience with popular NLP libraries and techniques. The project focuses on French-English text processing and demonstrates real-world applications of text mining methodologies.

## ✨ Features

### 🔤 Atelier 1: Translation Assistant & Document Similarity
- **NLTK-based Translation Assistant**: Interactive tool for English-to-French translation using semantic analysis
- **POS Tagging & Word Sense Disambiguation**: Uses Lesk algorithm for context-aware word meaning detection
- **Document Similarity Analysis**: Compares manual vs automatic translations using multiple similarity metrics:
  - Jaro-Winkler syntactic similarity
  - Jaccard coefficient on sentence pairs
  - Wu-Palmer semantic similarity via WordNet

### 🔍 Atelier 2: Advanced Plagiarism Detection System
- **Multi-representation Text Vectorization**: Implements 6 different text representation methods:
  - One-Hot Vector (OHV)
  - Bag-of-Words (BOW)
  - TF-IDF
  - Singular Value Decomposition (SVD/LSA)
  - Simple Embeddings
  - Character-level representation
- **Comprehensive Plagiarism Analysis**: Detects and classifies plagiarism types (clear plagiarism, reformulation, light synonymy)
- **Performance Evaluation System**: Scoring system with detailed comparative analysis of detection methods

### 📊 Atelier 3: Text Classification on Reuters Dataset
- **Multiple Vectorization Techniques**: BOW, One-Hot Encoding, Word2Vec, Doc2Vec
- **Machine Learning Classification**: Implementation of various classifiers:
  - K-Nearest Neighbors (KNN)
  - Random Forest (ensemble method)
  - Multi-layer Perceptron (MLP)
- **Feature Selection**: SelectKBest with chi-square test for optimal feature selection
- **Performance Analysis**: Comparative evaluation of different classification approaches

## 📁 Project Structure

```
Text-Mining/
├── Atelier_1.ipynb          # Translation assistance & similarity analysis
├── Atelier_2.ipynb          # Plagiarism detection system
├── Atelier_3.ipynb          # Text classification with Reuters dataset
└── README.md                 # Project documentation
```

## 🛠️ Tech Stack

### Core Libraries
- **NLTK** - Natural Language Toolkit for tokenization, POS tagging, and WordNet
- **scikit-learn** - Machine learning algorithms and text vectorization
- **NumPy** - Numerical computations and array operations
- **pandas** - Data manipulation and analysis

### Specialized Libraries
- **Gensim** - Word2Vec and Doc2Vec implementations
- **googletrans** - Google Translate API integration
- **jaro-winkler** - String similarity metrics
- **matplotlib** - Data visualization and plotting

### NLP Resources
- **WordNet** - Lexical database for semantic analysis
- **Reuters Corpus** - Standard dataset for text classification
- **NLTK Stopwords** - Language-specific stopword collections

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Internet connection (for downloading NLTK corpora)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/YASSIRAMRAOUI/Text-Mining.git
cd Text-Mining
```

2. **Create virtual environment** (recommended)
```bash
python -m venv text-mining-env
source text-mining-env/bin/activate  # On Windows: text-mining-env\Scripts\activate
```

3. **Install required packages**
```bash
pip install nltk scikit-learn numpy pandas gensim googletrans==4.0.0-rc1 jaro-winkler matplotlib
```

4. **Download NLTK resources**
```python
import nltk
nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4', 'stopwords', 'punkt_tab', 'reuters'])
```

## 💻 Usage

### Running the Workshops

1. **Start Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open desired workshop**
   - `Atelier_1.ipynb` - For translation assistance and similarity analysis
   - `Atelier_2.ipynb` - For plagiarism detection experiments
   - `Atelier_3.ipynb` - For text classification with Reuters dataset

3. **Execute cells sequentially** - Each notebook is designed to run from top to bottom

### Workshop-Specific Usage

#### Atelier 1: Translation Assistant
```python
# Interactive input required
text = input("Enter an English paragraph (minimum 3 sentences):")
manual_translation = input("Enter manual French translation:")
# System will analyze and compare translations
```

#### Atelier 2: Plagiarism Detection
```python
# Predefined corpus with plagiarism examples
# Run main() function to execute complete analysis
main()
```

#### Atelier 3: Text Classification
```python
# Uses Reuters dataset automatically
# Multiple classification methods compared
# Feature selection analysis included
```

## 📊 Example Output

### Translation Assistant Results
```
=== Translation Assistant ===
Sentence: "Artificial intelligence is a field of computer science."
- Word: intelligence
  > Meaning (EN): the ability to acquire and apply knowledge
  > French Translation(s): intelligence, intellect

Similarity Scores:
- Syntactic (Jaro-Winkler): 0.847
- Semantic (Wu-Palmer): 0.732
```

### Plagiarism Detection Results
```
COMPARATIVE PERFORMANCE TABLE
Method          | Total Score | Perfect Detections | Performance (%)
TFIDF          | 11/12       | 4/4               | 91.7%
SVD            | 10/12       | 4/4               | 83.3%
BOW            | 9/12        | 4/4               | 75.0%
```

## ⚙️ Configuration

### NLTK Data Path
If you encounter NLTK data errors, set the data path:
```python
import nltk
nltk.data.path.append('/path/to/nltk_data')
```

### Google Translate API
The project uses the free googletrans library. For production use, consider:
- Google Cloud Translation API
- Rate limiting implementation
- Error handling for API failures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new text analysis method'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include test cases for new features
- Update README for significant changes

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **NLTK Team** - For comprehensive NLP toolkit
- **scikit-learn Contributors** - For machine learning implementations  
- **Reuters Corpus** - For providing standard text classification dataset
- **Academic Community** - For NLP research and methodologies

---

**Note**: This project is designed for educational purposes and demonstrates various NLP techniques. For production applications, consider additional preprocessing, error handling, and performance optimizations." 
