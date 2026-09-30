"""
BERT / RoBERTa Sentiment Analysis Module.
Uses the specialized social media model 'cardiffnlp/twitter-roberta-base-sentiment'
via HuggingFace Transformers (PyTorch backend) for real 3-class inference (Positive, Negative, Neutral).
Computes confidence, distributions, temporal trends, platform breakdowns, and evaluation metrics if labels exist.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

LABEL_MAPPING = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "neg": "Negative",
    "neu": "Neutral",
    "pos": "Positive",
    "0": "Negative",
    "1": "Neutral",
    "2": "Positive"
}


class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading Sentiment Model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Sentiment Model loaded successfully.")

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """
        Runs real PyTorch inference on a list of texts in batches.
        Returns a list of dicts with 'sentiment' and 'confidence'.
        """
        results = []
        if not texts:
            return results

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # Replace empty text with placeholder space so tokenizer doesn't crash
            safe_texts = [t if (isinstance(t, str) and t.strip()) else " " for t in batch_texts]

            inputs = self.tokenizer(
                safe_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            for probs in probabilities:
                pred_idx = int(np.argmax(probs))
                raw_label = f"LABEL_{pred_idx}"
                sentiment = LABEL_MAPPING.get(raw_label, "Neutral")
                confidence = float(probs[pred_idx])

                results.append({
                    "sentiment": sentiment,
                    "confidence": round(confidence, 4),
                    "negative_score": round(float(probs[0]), 4),
                    "neutral_score": round(float(probs[1]), 4),
                    "positive_score": round(float(probs[2]), 4)
                })

        return results

    def compute_analytics(
        self,
        df: pd.DataFrame,
        ground_truth_col: Optional[str] = None
    ) -> dict:
        """
        Computes sentiment distribution, confidence distribution,
        sentiment over time, platform breakdown, and optional evaluation metrics.
        """
        total = len(df)
        if total == 0:
            return {
                "total_posts": 0,
                "distribution": {"Positive": 0, "Negative": 0, "Neutral": 0},
                "percentages": {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0},
                "average_confidence": 0.0,
                "confidence_distribution": [],
                "sentiment_over_time": [],
                "sentiment_by_platform": [],
                "evaluation": None
            }

        counts = df['sentiment'].value_counts().to_dict()
        pos_count = counts.get("Positive", 0)
        neg_count = counts.get("Negative", 0)
        neu_count = counts.get("Neutral", 0)

        distribution = {
            "Positive": int(pos_count),
            "Negative": int(neg_count),
            "Neutral": int(neu_count)
        }

        percentages = {
            "Positive": round((pos_count / total) * 100, 1),
            "Negative": round((neg_count / total) * 100, 1),
            "Neutral": round((neu_count / total) * 100, 1)
        }

        avg_confidence = round(float(df['confidence'].mean()), 4) if 'confidence' in df else 0.0

        # Confidence distribution buckets
        conf_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        conf_labels = ["< 50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        conf_series = pd.cut(df['confidence'], bins=conf_bins, labels=conf_labels, right=False)
        conf_dist_counts = conf_series.value_counts()[conf_labels].to_dict()
        confidence_distribution = [
            {"range": k, "count": int(v)} for k, v in conf_dist_counts.items()
        ]

        # Sentiment over time (if date available)
        sentiment_over_time = []
        if 'date_formatted' in df.columns and df['date_formatted'].notnull().any():
            valid_dates_df = df[df['date_formatted'].notnull()]
            if not valid_dates_df.empty:
                time_grouped = valid_dates_df.groupby(['date_formatted', 'sentiment']).size().unstack(fill_value=0)
                # Sort dates
                time_grouped = time_grouped.sort_index()
                for date_str, row in time_grouped.iterrows():
                    sentiment_over_time.append({
                        "date": str(date_str),
                        "Positive": int(row.get("Positive", 0)),
                        "Negative": int(row.get("Negative", 0)),
                        "Neutral": int(row.get("Neutral", 0)),
                        "Total": int(row.sum())
                    })

        # Sentiment by platform (if platform available)
        sentiment_by_platform = []
        if 'platform' in df.columns and (df['platform'] != 'General').any():
            plat_grouped = df.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
            for plat, row in plat_grouped.iterrows():
                sentiment_by_platform.append({
                    "platform": str(plat),
                    "Positive": int(row.get("Positive", 0)),
                    "Negative": int(row.get("Negative", 0)),
                    "Neutral": int(row.get("Neutral", 0)),
                    "Total": int(row.sum())
                })

        # Evaluation metrics if ground truth label provided
        evaluation = None
        if ground_truth_col and ground_truth_col in df.columns:
            try:
                # Normalize ground truth labels
                gt_raw = df[ground_truth_col].astype(str).str.strip().str.lower()
                gt_normalized = gt_raw.map(lambda x: LABEL_MAPPING.get(x, None))
                
                valid_mask = gt_normalized.notnull()
                if valid_mask.sum() > 0:
                    y_true = gt_normalized[valid_mask]
                    y_pred = df.loc[valid_mask, 'sentiment']

                    labels = ["Negative", "Neutral", "Positive"]
                    acc = float(accuracy_score(y_true, y_pred))
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, labels=labels, average="weighted", zero_division=0
                    )
                    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

                    evaluation = {
                        "accuracy": round(acc, 4),
                        "precision": round(float(prec), 4),
                        "recall": round(float(rec), 4),
                        "f1_score": round(float(f1), 4),
                        "evaluated_samples": int(valid_mask.sum()),
                        "labels": labels,
                        "confusion_matrix": cm
                    }
            except Exception as e:
                print(f"Error computing evaluation metrics: {e}")

        return {
            "total_posts": total,
            "distribution": distribution,
            "percentages": percentages,
            "average_confidence": avg_confidence,
            "confidence_distribution": confidence_distribution,
            "sentiment_over_time": sentiment_over_time,
            "sentiment_by_platform": sentiment_by_platform,
            "evaluation": evaluation
        }
