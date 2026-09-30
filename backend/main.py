"""
FastAPI Application for AI-Powered Social Media Sentiment and Trend Analysis.
Provides endpoints for dataset upload validation, column auto-detection,
NLP preprocessing, real BERT sentiment analysis, LDA topic modeling,
trend aggregation, and Power BI dataset export.
"""

import os
import sys
import io
import uuid
import json
import logging
from typing import Optional

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from preprocessor import preprocess_dataset
from sentiment_analyzer import SentimentAnalyzer
from topic_modeler import TopicModeler
from hashtag_analyzer import analyze_hashtags
from exporter import (
    prepare_powerbi_dataframe,
    export_to_csv_bytes,
    export_to_excel_bytes,
    get_powerbi_schema_metadata
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SocialMediaDashboardAPI")

app = FastAPI(
    title="Social Media Sentiment & Trend Analysis API",
    description="Real BERT & LDA analysis engine for user-provided social media datasets",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP Models (Cached RoBERTa and LDA)
sentiment_analyzer = SentimentAnalyzer()
topic_modeler = TopicModeler(default_num_topics=5)

# In-memory storage for user session datasets (avoids requiring database)
# Key: session_id -> dict with 'raw_df', 'processed_df', 'pbi_df', 'summary'
session_store = {}

# Common variations of column names
COLUMN_CANDIDATES = {
    "text": ["text", "tweet", "content", "post", "comment", "message", "body", "review", "description", "status", "clean_text", "caption"],
    "id": ["post_id", "id", "tweet_id", "comment_id", "uid", "identifier", "item_id"],
    "date": ["date", "created_at", "timestamp", "datetime", "time", "published_at", "post_date", "created"],
    "platform": ["platform", "source", "channel", "network", "site", "social_media"],
    "hashtags": ["hashtags", "tags", "hashtag", "tag"],
    "sentiment": ["sentiment", "label", "target", "sentiment_label", "ground_truth", "actual_sentiment"]
}


def detect_columns(columns: list[str]) -> dict:
    """
    Automatically detects best-match column names from available dataset columns.
    """
    detected = {
        "text_col": None,
        "id_col": None,
        "date_col": None,
        "platform_col": None,
        "hashtags_col": None,
        "sentiment_col": None
    }
    col_lower_map = {str(c).lower().strip(): c for c in columns}

    for col_type, candidates in COLUMN_CANDIDATES.items():
        # First check exact matches
        for cand in candidates:
            if cand in col_lower_map:
                detected[f"{col_type}_col"] = col_lower_map[cand]
                break
        # If no exact match, check partial containment
        if detected[f"{col_type}_col"] is None:
            for c_lower, orig_col in col_lower_map.items():
                if any(cand in c_lower for cand in candidates):
                    detected[f"{col_type}_col"] = orig_col
                    break

    return detected


def parse_uploaded_file(contents: bytes, filename: str) -> pd.DataFrame:
    """
    Parses CSV, Excel, or JSON bytes into a pandas DataFrame.
    """
    fn_lower = filename.lower()
    try:
        if fn_lower.endswith('.csv') or fn_lower.endswith('.txt'):
            # Try standard utf-8 then latin-1
            try:
                return pd.read_csv(io.BytesIO(contents), encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        elif fn_lower.endswith('.xlsx') or fn_lower.endswith('.xls'):
            return pd.read_excel(io.BytesIO(contents))
        elif fn_lower.endswith('.json'):
            try:
                # Try orient='records' or standard json
                return pd.read_json(io.BytesIO(contents))
            except Exception:
                data = json.loads(contents.decode('utf-8'))
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Check if items or records are inside a top-level key
                    for k in ['data', 'posts', 'records', 'items', 'tweets']:
                        if k in data and isinstance(data[k], list):
                            return pd.DataFrame(data[k])
                    return pd.DataFrame([data])
                else:
                    raise ValueError("Unsupported JSON format")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{filename}'. Please upload a CSV, Excel (.xlsx), or JSON file."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse '{filename}': {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "model": sentiment_analyzer.model_name,
        "device": str(sentiment_analyzer.device),
        "version": "2.0.0"
    }


@app.post("/api/upload-preview")
async def upload_preview(file: UploadFile = File(...)):
    """
    Upload Dataset -> Validate File -> Detect Columns -> Preview Data
    Accepts CSV, Excel, JSON. Returns first rows, detected columns, and dataset stats.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    file_size_bytes = len(contents)
    file_size_formatted = f"{file_size_bytes / (1024 * 1024):.2f} MB" if file_size_bytes > 1024*1024 else f"{file_size_bytes / 1024:.1f} KB"

    df = parse_uploaded_file(contents, file.filename)
    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded dataset contains zero records.")

    columns = [str(c) for c in df.columns]
    detected_cols = detect_columns(columns)

    # Basic stats
    total_rows = len(df)
    null_counts = {str(c): int(df[c].isnull().sum()) for c in columns}

    # Store raw dataframe in session
    session_id = str(uuid.uuid4())
    session_store[session_id] = {
        "filename": file.filename,
        "file_size": file_size_formatted,
        "raw_df": df
    }

    # Convert first 10 rows to preview records (converting NaNs to None for JSON serialization)
    preview_df = df.head(10).copy()
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    preview_records = preview_df.to_dict(orient="records")

    validation_errors = []
    if not detected_cols["text_col"]:
        validation_errors.append(
            "Primary text column could not be automatically detected. Please select the appropriate text column manually."
        )

    return {
        "session_id": session_id,
        "filename": file.filename,
        "file_size": file_size_formatted,
        "total_rows": total_rows,
        "columns": columns,
        "detected_columns": detected_cols,
        "null_counts": null_counts,
        "preview_records": preview_records,
        "validation_errors": validation_errors,
        "is_valid": len(columns) > 0
    }


@app.post("/api/analyze")
async def analyze_dataset(
    session_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text_col: str = Form(...),
    id_col: Optional[str] = Form(None),
    date_col: Optional[str] = Form(None),
    platform_col: Optional[str] = Form(None),
    hashtags_col: Optional[str] = Form(None),
    sentiment_col: Optional[str] = Form(None),
    num_topics: Optional[int] = Form(5),
    remove_duplicates: Optional[bool] = Form(True)
):
    """
    Executes full pipeline:
    Validate -> Preprocess -> BERT Sentiment -> LDA Topics -> Trend Analysis -> Power BI Dataset
    """
    df = None
    filename = "uploaded_dataset"

    if session_id and session_id in session_store and "raw_df" in session_store[session_id]:
        df = session_store[session_id]["raw_df"].copy()
        filename = session_store[session_id].get("filename", filename)
    elif file is not None:
        contents = await file.read()
        filename = file.filename
        df = parse_uploaded_file(contents, filename)
    else:
        raise HTTPException(
            status_code=400,
            detail="No valid dataset provided. Please upload a file or pass an active session_id."
        )

    if text_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Selected text column '{text_col}' was not found in dataset columns: {list(df.columns)}"
        )

    # 1. NLP Preprocessing
    logger.info(f"Starting NLP preprocessing on {len(df)} rows using text column '{text_col}'...")
    processed_df, prep_stats = preprocess_dataset(
        df=df,
        text_col=text_col,
        id_col=id_col if id_col in df.columns else None,
        date_col=date_col if date_col in df.columns else None,
        platform_col=platform_col if platform_col in df.columns else None,
        hashtags_col=hashtags_col if hashtags_col in df.columns else None,
        remove_duplicates=bool(remove_duplicates)
    )

    if processed_df.empty:
        raise HTTPException(
            status_code=400,
            detail="All records were dropped during preprocessing (empty or invalid text). Please check your dataset."
        )

    # 2. Real BERT Sentiment Analysis
    logger.info(f"Running BERT sentiment analysis on {len(processed_df)} records...")
    # Use bert_text (cleaned for BERT with preserved grammar and emojis)
    texts_for_bert = processed_df['bert_text'].tolist()
    bert_predictions = sentiment_analyzer.analyze_batch(texts_for_bert, batch_size=32)

    processed_df['sentiment'] = [p['sentiment'] for p in bert_predictions]
    processed_df['confidence'] = [p['confidence'] for p in bert_predictions]

    # Compute sentiment analytics
    sentiment_analytics = sentiment_analyzer.compute_analytics(
        df=processed_df,
        ground_truth_col=sentiment_col if sentiment_col in df.columns else None
    )

    # 3. LDA Topic Modeling
    logger.info("Executing LDA topic modeling...")
    processed_df, topic_metadata = topic_modeler.fit_transform(
        df=processed_df,
        cleaned_text_col='cleaned_text',
        num_topics=num_topics or 5
    )

    # 4. Hashtag Analysis
    logger.info("Analyzing hashtags...")
    hashtag_data = analyze_hashtags(processed_df)

    # 5. Prepare Power BI Dataset
    logger.info("Preparing Power BI output schema...")
    pbi_df = prepare_powerbi_dataframe(processed_df)

    # Generate or refresh session ID
    if not session_id:
        session_id = str(uuid.uuid4())

    session_store[session_id] = {
        "filename": filename,
        "processed_df": processed_df,
        "pbi_df": pbi_df
    }

    # Top predictions sample (top 100 records for UI table)
    sample_columns = ['post_id', 'original_text', 'sentiment', 'confidence', 'topic_label']
    if prep_stats['has_date']:
        sample_columns.append('date_formatted')
    if prep_stats['has_platform']:
        sample_columns.append('platform')
    if prep_stats['has_hashtags']:
        sample_columns.append('hashtags')

    sample_slice = processed_df.head(100)[sample_columns].copy()
    sample_slice = sample_slice.where(pd.notnull(sample_slice), None)
    # Ensure hashtags are list of strings
    if 'hashtags' in sample_slice.columns:
        sample_slice['hashtags'] = sample_slice['hashtags'].apply(
            lambda x: list(x) if isinstance(x, (list, set)) else []
        )
    sample_predictions = sample_slice.to_dict(orient="records")

    # Available filter values
    available_filters = {
        "sentiments": ["Positive", "Neutral", "Negative"],
        "platforms": sorted(list(processed_df['platform'].unique())) if prep_stats['has_platform'] else [],
        "topics": [{"id": t["topic_id"], "label": t["label"]} for t in topic_metadata["topics"]],
        "has_date": prep_stats['has_date'],
        "has_platform": prep_stats['has_platform'],
        "has_hashtags": hashtag_data['available'],
        "date_min": str(processed_df['date_formatted'].min()) if prep_stats['has_date'] and processed_df['date_formatted'].notnull().any() else None,
        "date_max": str(processed_df['date_formatted'].max()) if prep_stats['has_date'] and processed_df['date_formatted'].notnull().any() else None
    }

    powerbi_metadata = get_powerbi_schema_metadata()

    return {
        "session_id": session_id,
        "filename": filename,
        "preprocessing_stats": prep_stats,
        "sentiment_analytics": sentiment_analytics,
        "topic_metadata": topic_metadata,
        "hashtag_analytics": hashtag_data,
        "sample_predictions": sample_predictions,
        "available_filters": available_filters,
        "powerbi_metadata": powerbi_metadata
    }


@app.get("/api/export/powerbi")
async def export_powerbi(
    session_id: str = Query(...),
    format: str = Query("csv", regex="^(csv|xlsx)$")
):
    """
    Downloads the Power BI ready dataset as CSV or Excel (.xlsx).
    """
    if session_id not in session_store or "pbi_df" not in session_store[session_id]:
        raise HTTPException(
            status_code=404,
            detail="Session expired or dataset not found. Please re-run analysis."
        )

    pbi_df = session_store[session_id]["pbi_df"]
    base_name = os.path.splitext(session_store[session_id].get("filename", "social_media_analytics"))[0]

    if format == "csv":
        data_bytes = export_to_csv_bytes(pbi_df)
        filename = f"{base_name}_PowerBI_Export.csv"
        media_type = "text/csv; charset=utf-8"
    else:
        data_bytes = export_to_excel_bytes(pbi_df)
        filename = f"{base_name}_PowerBI_Export.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    return StreamingResponse(
        io.BytesIO(data_bytes),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
