import nltk
import os

# Set the NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Try to load stopwords; if not available, run the download script
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    import download_nltk_data  # This will run the download script
