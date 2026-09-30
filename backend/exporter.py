"""
Power BI Dataset Exporter & Integration Module.
Prepares clean, enriched social media analytics datasets formatted for Power BI.
Exports columns:
post_id, original_text, cleaned_text, sentiment, confidence, topic_id,
topic_keywords, date, platform, hashtags, entities.
Supports direct CSV and Excel (.xlsx) downloads, plus DAX measures and schema guides.
"""

import io
import re
import pandas as pd
from typing import Optional

def extract_basic_entities(text: str) -> str:
    """
    Extracts high-signal entities (e.g., proper nouns, acronyms, capitalized multi-word phrases)
    without requiring heavy external NER models.
    """
    if not isinstance(text, str):
        return ""
    # Matches capitalized words and phrases (e.g. "OpenAI", "Apple iPhone", "New York")
    entity_pattern = re.compile(r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b')
    matches = entity_pattern.findall(text)
    # Filter common sentence-starters or noise
    filtered = [m for m in matches if len(m) > 2 and m.lower() not in {'the', 'this', 'that', 'with', 'from'}]
    # Remove duplicates preserving order
    seen = set()
    unique_entities = []
    for item in filtered:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique_entities.append(item)
    return "; ".join(unique_entities[:10])


def prepare_powerbi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the exact schema for Power BI consumption.
    """
    pbi_df = pd.DataFrame()

    pbi_df['post_id'] = df['post_id'] if 'post_id' in df.columns else [f"P_{i+1:05d}" for i in range(len(df))]
    pbi_df['original_text'] = df['original_text'] if 'original_text' in df.columns else ""
    pbi_df['cleaned_text'] = df['cleaned_text'] if 'cleaned_text' in df.columns else ""
    pbi_df['sentiment'] = df['sentiment'] if 'sentiment' in df.columns else "Neutral"
    pbi_df['confidence'] = df['confidence'] if 'confidence' in df.columns else 0.0
    pbi_df['topic_id'] = df['topic_id'] if 'topic_id' in df.columns else 0
    pbi_df['topic_keywords'] = df['topic_keywords'] if 'topic_keywords' in df.columns else ""
    
    # Date
    if 'date_formatted' in df.columns and df['date_formatted'].notnull().any():
        pbi_df['date'] = df['date_formatted']
    elif 'date' in df.columns and df['date'].notnull().any():
        pbi_df['date'] = df['date'].astype(str)
    else:
        pbi_df['date'] = ""

    # Platform
    pbi_df['platform'] = df['platform'] if 'platform' in df.columns else "General"

    # Hashtags formatted as string
    if 'hashtags' in df.columns:
        pbi_df['hashtags'] = df['hashtags'].apply(
            lambda x: "; ".join([f"#{t}" if not str(t).startswith('#') else str(t) for t in x])
            if isinstance(x, (list, set)) else str(x or "")
        )
    else:
        pbi_df['hashtags'] = ""

    # Entities
    if 'original_text' in df.columns:
        pbi_df['entities'] = df['original_text'].apply(extract_basic_entities)
    else:
        pbi_df['entities'] = ""

    return pbi_df


def export_to_csv_bytes(pbi_df: pd.DataFrame) -> bytes:
    """Exports DataFrame to UTF-8 CSV bytes with BOM for flawless Excel/Power BI ingestion."""
    buffer = io.StringIO()
    pbi_df.to_csv(buffer, index=False, encoding='utf-8')
    return buffer.getvalue().encode('utf-8-sig')


def export_to_excel_bytes(pbi_df: pd.DataFrame) -> bytes:
    """Exports DataFrame to Excel (.xlsx) bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pbi_df.to_excel(writer, index=False, sheet_name="Social_Media_Analytics")
    return output.getvalue()


def get_powerbi_schema_metadata() -> dict:
    """
    Returns schema metadata, column definitions, and suggested DAX formulas.
    """
    return {
        "columns": [
            {"name": "post_id", "type": "Text", "description": "Unique identifier for each post/comment"},
            {"name": "original_text", "type": "Text", "description": "Raw input text as published"},
            {"name": "cleaned_text", "type": "Text", "description": "Preprocessed text used for topic modeling and keyword extraction"},
            {"name": "sentiment", "type": "Text", "description": "Predicted sentiment: Positive, Negative, or Neutral"},
            {"name": "confidence", "type": "Decimal", "description": "Model prediction confidence score (0.0 to 1.0)"},
            {"name": "topic_id", "type": "Whole Number", "description": "Dominant LDA topic cluster ID"},
            {"name": "topic_keywords", "type": "Text", "description": "Key topical terms representing this cluster"},
            {"name": "date", "type": "Date", "description": "Timestamp/Date of publication (if available)"},
            {"name": "platform", "type": "Text", "description": "Origin social media platform or channel"},
            {"name": "hashtags", "type": "Text", "description": "Associated hashtags separated by semicolon"},
            {"name": "entities", "type": "Text", "description": "Extracted proper nouns and high-signal entities"}
        ],
        "dax_measures": [
            {
                "name": "Net Sentiment Score (NSS)",
                "formula": "DIVIDE(CALCULATE(COUNTROWS('Social_Media_Analytics'), 'Social_Media_Analytics'[sentiment] = \"Positive\") - CALCULATE(COUNTROWS('Social_Media_Analytics'), 'Social_Media_Analytics'[sentiment] = \"Negative\"), COUNTROWS('Social_Media_Analytics'), 0) * 100"
            },
            {
                "name": "Positive Ratio %",
                "formula": "DIVIDE(CALCULATE(COUNTROWS('Social_Media_Analytics'), 'Social_Media_Analytics'[sentiment] = \"Positive\"), COUNTROWS('Social_Media_Analytics'), 0)"
            },
            {
                "name": "Average Model Confidence",
                "formula": "AVERAGE('Social_Media_Analytics'[confidence])"
            }
        ]
    }
