"""
NLP Preprocessing Pipeline for Social Media Sentiment & Trend Analysis.
Handles missing values, duplicate removal, text cleaning, emoji normalization,
tokenization, stopword removal, and lemmatization.
Preserves both original_text (for BERT context) and cleaned_text (for LDA and keyword analytics).
"""

import re
import html
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is available
try:
    _stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    _stopwords = set(stopwords.words('english'))

try:
    _lemmatizer = WordNetLemmatizer()
    _lemmatizer.lemmatize("testing")
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    _lemmatizer = WordNetLemmatizer()

# Extended stop words common in social media noise
SOCIAL_STOPWORDS = {
    'rt', 'via', 'amp', 'u', 'ur', 'im', 'dont', 'cant', 'didnt', 'wont',
    'thats', 'youre', 'theyre', 'weve', 'couldnt', 'wouldnt', 'shouldnt',
    'http', 'https', 'co', 'com', 'www', 'get', 'like', 'one', 'would',
    'also', 'know', 'see', 'us', 'say', 'go', 'make', 'even', 'think'
}
ALL_STOPWORDS = _stopwords.union(SOCIAL_STOPWORDS)

# URL and Mention regex
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_REGEX = re.compile(r'<.*?>')
MENTION_REGEX = re.compile(r'@\w+')
HASHTAG_REGEX = re.compile(r'#(\w+)')
WHITESPACE_REGEX = re.compile(r'\s+')
NON_ALPHANUM_REGEX = re.compile(r'[^a-zA-Z0-9\s]')


def extract_hashtags_from_text(text: str) -> list[str]:
    """Extract hashtag words from text without the '#' symbol."""
    if not isinstance(text, str):
        return []
    matches = HASHTAG_REGEX.findall(text)
    return [m.lower() for m in matches if len(m) > 1]


def clean_text_for_bert(text: str) -> str:
    """
    Light preprocessing for BERT to preserve grammatical structure,
    punctuation sentiment cues, emojis, and casing where helpful,
    while removing noisy URLs, HTML tags, and normalizing excessive whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Decode HTML entities like &amp;, &lt;
    t = html.unescape(text)
    
    # Remove URLs
    t = URL_REGEX.sub('', t)
    
    # Remove HTML tags
    t = HTML_TAG_REGEX.sub('', t)
    
    # Normalize user mentions to @user (standard for twitter roberta)
    t = MENTION_REGEX.sub('@user', t)
    
    # Normalize whitespace
    t = WHITESPACE_REGEX.sub(' ', t).strip()
    return t


def clean_text_for_lda(text: str) -> str:
    """
    Deeper preprocessing for Topic Modeling (LDA) and keyword analytics:
    lowercasing, punctuation removal, stopword removal, and lemmatization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Decode HTML and remove URLs
    t = html.unescape(text)
    t = URL_REGEX.sub('', t)
    t = HTML_TAG_REGEX.sub('', t)
    t = MENTION_REGEX.sub('', t)
    
    # Lowercase
    t = t.lower()
    
    # Keep only letters and whitespace
    t = re.sub(r'[^a-z\s]', ' ', t)
    
    # Tokenize by whitespace (robust and fast)
    tokens = t.split()
    
    # Lemmatize and filter stopwords & single characters
    cleaned_tokens = [
        _lemmatizer.lemmatize(word)
        for word in tokens
        if word not in ALL_STOPWORDS and len(word) > 2
    ]
    
    return " ".join(cleaned_tokens)


def preprocess_dataset(
    df: pd.DataFrame,
    text_col: str,
    id_col: str = None,
    date_col: str = None,
    platform_col: str = None,
    hashtags_col: str = None,
    remove_duplicates: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Executes the NLP preprocessing pipeline on the input DataFrame.
    
    Returns:
        processed_df: Enriched dataframe containing original_text, cleaned_text,
                      normalized columns, and extracted features.
        stats: Dictionary of preprocessing statistics (dropped nulls, duplicates, row count).
    """
    initial_count = len(df)
    
    # 1. Handle missing values in primary text column
    working_df = df.copy()
    working_df = working_df.dropna(subset=[text_col])
    working_df = working_df[working_df[text_col].astype(str).str.strip() != '']
    dropped_empty_text = initial_count - len(working_df)
    
    # 2. Handle duplicate records
    dropped_duplicates = 0
    if remove_duplicates:
        count_before_dedup = len(working_df)
        working_df = working_df.drop_duplicates(subset=[text_col])
        dropped_duplicates = count_before_dedup - len(working_df)
    
    # Reset index
    working_df = working_df.reset_index(drop=True)
    
    # 3. Standardize Post ID
    if id_col and id_col in working_df.columns:
        working_df['post_id'] = working_df[id_col].astype(str)
    else:
        working_df['post_id'] = [f"POST_{i+1:05d}" for i in range(len(working_df))]
    
    # 4. Text columns: Keep original_text and generate cleaned_text
    working_df['original_text'] = working_df[text_col].astype(str).str.strip()
    working_df['bert_text'] = working_df['original_text'].apply(clean_text_for_bert)
    working_df['cleaned_text'] = working_df['original_text'].apply(clean_text_for_lda)
    
    # 5. Standardize Date column if provided
    has_date = False
    if date_col and date_col in working_df.columns:
        try:
            working_df['date'] = pd.to_datetime(working_df[date_col], errors='coerce')
            # Check if at least some dates are valid
            valid_date_count = working_df['date'].notnull().sum()
            if valid_date_count > 0:
                has_date = True
                working_df['date_formatted'] = working_df['date'].dt.strftime('%Y-%m-%d')
            else:
                working_df['date'] = None
                working_df['date_formatted'] = None
        except Exception:
            working_df['date'] = None
            working_df['date_formatted'] = None
    else:
        working_df['date'] = None
        working_df['date_formatted'] = None
    
    # 6. Standardize Platform column if provided
    has_platform = False
    if platform_col and platform_col in working_df.columns:
        working_df['platform'] = working_df[platform_col].astype(str).str.strip()
        working_df['platform'] = working_df['platform'].replace(['nan', 'None', ''], 'Unknown')
        has_platform = True
    else:
        working_df['platform'] = 'General'
    
    # 7. Standardize Hashtags column if provided, or extract from text
    has_hashtags = False
    if hashtags_col and hashtags_col in working_df.columns:
        def parse_existing_hashtags(val):
            if pd.isna(val):
                return []
            if isinstance(val, list):
                return [str(v).replace('#', '').strip().lower() for v in val if str(v).strip()]
            val_str = str(val).strip()
            cleaned = re.sub(r'[\[\]\'"]', '', val_str)
            parts = re.split(r'[,;|\s]+', cleaned)
            return [p.replace('#', '').strip().lower() for p in parts if p.replace('#', '').strip()]
        
        working_df['hashtags'] = working_df[hashtags_col].apply(parse_existing_hashtags)
        if working_df['hashtags'].apply(len).sum() > 0:
            has_hashtags = True
    else:
        # Auto-extract from original text
        working_df['hashtags'] = working_df['original_text'].apply(extract_hashtags_from_text)
        if working_df['hashtags'].apply(len).sum() > 0:
            has_hashtags = True

    stats = {
        "initial_rows": initial_count,
        "valid_rows": len(working_df),
        "dropped_empty_rows": dropped_empty_text,
        "dropped_duplicate_rows": dropped_duplicates,
        "has_date": has_date,
        "has_platform": has_platform,
        "has_hashtags": has_hashtags
    }
    
    return working_df, stats
