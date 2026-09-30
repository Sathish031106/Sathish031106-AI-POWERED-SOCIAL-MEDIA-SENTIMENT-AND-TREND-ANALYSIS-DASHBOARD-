"""
Hashtag Analytics Module.
Analyzes hashtag frequencies, sentiment associations, and temporal trends.
Gracefully indicates when hashtags are unavailable in the dataset.
"""

import pandas as pd
from collections import Counter
from typing import Optional


def analyze_hashtags(df: pd.DataFrame) -> dict:
    """
    Computes hashtag frequency, sentiment correlation, and temporal trends.
    """
    if 'hashtags' not in df.columns:
        return {
            "available": False,
            "message": "No hashtag data or hashtag patterns detected in this dataset.",
            "top_hashtags": [],
            "hashtag_trends": [],
            "hashtag_sentiment": []
        }

    # Flatten hashtags with their associated row sentiment and date
    records = []
    for idx, row in df.iterrows():
        tags = row.get('hashtags', [])
        if not isinstance(tags, list):
            continue
        sentiment = row.get('sentiment', 'Neutral')
        date_val = row.get('date_formatted', None)
        for t in tags:
            tag_clean = str(t).lower().replace('#', '').strip()
            if tag_clean and len(tag_clean) > 1:
                records.append({
                    "hashtag": f"#{tag_clean}",
                    "sentiment": sentiment,
                    "date": date_val
                })

    if not records:
        return {
            "available": False,
            "message": "No hashtags were detected in the uploaded records.",
            "top_hashtags": [],
            "hashtag_trends": [],
            "hashtag_sentiment": []
        }

    tag_df = pd.DataFrame(records)

    # 1. Top hashtags by frequency
    tag_counts = tag_df['hashtag'].value_counts()
    top_n = tag_counts.head(20)
    top_hashtags = [
        {"hashtag": tag, "count": int(count)}
        for tag, count in top_n.items()
    ]

    top_tag_names = set(top_n.index)

    # 2. Sentiment associated with top hashtags
    hashtag_sentiment = []
    top_tags_df = tag_df[tag_df['hashtag'].isin(top_tag_names)]
    sent_grouped = top_tags_df.groupby(['hashtag', 'sentiment']).size().unstack(fill_value=0)
    for tag in top_n.index:
        if tag in sent_grouped.index:
            row = sent_grouped.loc[tag]
            hashtag_sentiment.append({
                "hashtag": tag,
                "Positive": int(row.get("Positive", 0)),
                "Negative": int(row.get("Negative", 0)),
                "Neutral": int(row.get("Neutral", 0)),
                "Total": int(row.sum())
            })

    # 3. Hashtag trends over time (if date available)
    hashtag_trends = []
    if 'date' in tag_df.columns and tag_df['date'].notnull().any():
        valid_date_tags = tag_df[tag_df['date'].notnull() & tag_df['hashtag'].isin(list(top_n.index[:5]))]
        if not valid_date_tags.empty:
            time_tag = valid_date_tags.groupby(['date', 'hashtag']).size().unstack(fill_value=0)
            time_tag = time_tag.sort_index()
            for date_str, row in time_tag.iterrows():
                entry = {"date": str(date_str)}
                for tag in top_n.index[:5]:
                    entry[tag] = int(row.get(tag, 0))
                hashtag_trends.append(entry)

    return {
        "available": True,
        "message": f"Found {len(tag_counts)} unique hashtags across {len(records)} occurrences.",
        "unique_count": len(tag_counts),
        "total_occurrences": len(records),
        "top_hashtags": top_hashtags,
        "hashtag_sentiment": hashtag_sentiment,
        "hashtag_trends": hashtag_trends
    }
