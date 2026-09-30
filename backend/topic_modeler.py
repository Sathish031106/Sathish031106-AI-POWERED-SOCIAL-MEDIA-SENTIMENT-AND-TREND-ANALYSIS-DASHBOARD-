"""
LDA (Latent Dirichlet Allocation) Topic Modeling & Trend Analysis Module.
Converts cleaned and lemmatized social media text into a Document-Term Matrix (DTM),
discovers latent topics, extracts top keywords with weights, assigns dominant topics,
and computes topic distribution, sentiment by topic, and temporal topic trends.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TopicModeler:
    def __init__(self, default_num_topics: int = 5, top_words_count: int = 10):
        self.default_num_topics = default_num_topics
        self.top_words_count = top_words_count

    def fit_transform(
        self,
        df: pd.DataFrame,
        cleaned_text_col: str = 'cleaned_text',
        num_topics: Optional[int] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Fits LDA model on the cleaned_text column, assigns dominant topic to each row,
        and generates topic metadata, keyword distributions, and topic statistics.
        """
        texts = df[cleaned_text_col].fillna("").astype(str).tolist()
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]

        # If very few non-empty texts, handle gracefully
        if len(non_empty_indices) < 2:
            df['topic_id'] = 0
            df['topic_label'] = "General Discussion"
            df['topic_keywords'] = "social, media, posts"
            return df, {
                "num_topics": 1,
                "topics": [{
                    "topic_id": 0,
                    "label": "General Discussion",
                    "keywords": [{"word": "social", "weight": 1.0}],
                    "post_count": len(df),
                    "percentage": 100.0
                }],
                "top_keywords_overall": [],
                "topic_distribution": [{"topic_id": 0, "label": "General Discussion", "count": len(df)}],
                "topic_trends_over_time": [],
                "sentiment_by_topic": []
            }

        # Vectorize: Document-Term Matrix (DTM)
        # Filter terms that appear in at least 1 document, stop words already filtered but extra check
        min_df = 1
        max_df = 0.98 if len(non_empty_indices) > 20 else 1.0
        
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=2500,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )

        try:
            dtm = vectorizer.fit_transform(texts)
            vocab = vectorizer.get_feature_names_out()
        except ValueError:
            # Fallback if vocabulary was empty
            df['topic_id'] = 0
            df['topic_label'] = "Topic 0"
            df['topic_keywords'] = "post, update"
            return df, {
                "num_topics": 1,
                "topics": [{"topic_id": 0, "label": "Topic 0", "keywords": [], "post_count": len(df), "percentage": 100.0}],
                "top_keywords_overall": [],
                "topic_distribution": [{"topic_id": 0, "label": "Topic 0", "count": len(df)}],
                "topic_trends_over_time": [],
                "sentiment_by_topic": []
            }

        # Determine number of topics: bounded by vocabulary size and document count
        n_features = len(vocab)
        n_samples = len(texts)
        chosen_k = num_topics if num_topics and num_topics > 0 else self.default_num_topics
        chosen_k = min(chosen_k, n_samples, max(1, n_features // 2))
        chosen_k = max(1, chosen_k)

        # Fit Latent Dirichlet Allocation
        lda = LatentDirichletAllocation(
            n_components=chosen_k,
            random_state=42,
            learning_method='online',
            max_iter=15,
            batch_size=128
        )
        doc_topic_dist = lda.fit_transform(dtm)

        # Extract top keywords per topic
        topics_info = []
        topic_keyword_map = {}
        topic_label_map = {}

        for topic_idx, topic_weights in enumerate(lda.components_):
            top_kw_indices = topic_weights.argsort()[: -self.top_words_count - 1 : -1]
            total_topic_weight = float(np.sum(topic_weights[top_kw_indices])) + 1e-9
            
            keywords_list = []
            for idx in top_kw_indices:
                word = str(vocab[idx])
                raw_weight = float(topic_weights[idx])
                keywords_list.append({
                    "word": word,
                    "weight": round(raw_weight / total_topic_weight, 4)
                })

            top_3_words = [kw["word"] for kw in keywords_list[:3]]
            label = f"Topic {topic_idx + 1}: {', '.join(top_3_words)}" if top_3_words else f"Topic {topic_idx + 1}"
            keywords_str = ", ".join([kw["word"] for kw in keywords_list[:5]])

            topic_label_map[topic_idx] = label
            topic_keyword_map[topic_idx] = keywords_str

            topics_info.append({
                "topic_id": topic_idx,
                "label": label,
                "keywords": keywords_list,
                "top_keywords_str": keywords_str,
                "post_count": 0,
                "percentage": 0.0
            })

        # Assign dominant topic to each post
        dominant_topics = np.argmax(doc_topic_dist, axis=1)
        df['topic_id'] = dominant_topics
        df['topic_label'] = df['topic_id'].map(topic_label_map)
        df['topic_keywords'] = df['topic_id'].map(topic_keyword_map)

        # Count frequencies
        topic_counts = df['topic_id'].value_counts().to_dict()
        total_posts = len(df)
        for t in topics_info:
            c = topic_counts.get(t["topic_id"], 0)
            t["post_count"] = int(c)
            t["percentage"] = round((c / total_posts) * 100, 1) if total_posts > 0 else 0.0

        # Overall top keywords across entire corpus
        sum_dtm = np.asarray(dtm.sum(axis=0)).flatten()
        top_overall_indices = sum_dtm.argsort()[: -25 : -1]
        top_keywords_overall = [
            {"word": str(vocab[i]), "count": int(sum_dtm[i])}
            for i in top_overall_indices
            if sum_dtm[i] > 0
        ]

        # Topic trends over time (if date formatted available)
        topic_trends_over_time = []
        if 'date_formatted' in df.columns and df['date_formatted'].notnull().any():
            valid_dates_df = df[df['date_formatted'].notnull()]
            if not valid_dates_df.empty:
                time_topic = valid_dates_df.groupby(['date_formatted', 'topic_id']).size().unstack(fill_value=0)
                time_topic = time_topic.sort_index()
                for date_str, row in time_topic.iterrows():
                    entry = {"date": str(date_str)}
                    for tid in range(chosen_k):
                        entry[f"Topic_{tid}"] = int(row.get(tid, 0))
                    topic_trends_over_time.append(entry)

        # Sentiment by topic (cross-tabulation)
        sentiment_by_topic = []
        if 'sentiment' in df.columns:
            topic_sent = df.groupby(['topic_id', 'sentiment']).size().unstack(fill_value=0)
            for tid in range(chosen_k):
                label = topic_label_map.get(tid, f"Topic {tid}")
                if tid in topic_sent.index:
                    row = topic_sent.loc[tid]
                    sentiment_by_topic.append({
                        "topic_id": tid,
                        "label": label,
                        "Positive": int(row.get("Positive", 0)),
                        "Negative": int(row.get("Negative", 0)),
                        "Neutral": int(row.get("Neutral", 0)),
                        "Total": int(row.sum())
                    })
                else:
                    sentiment_by_topic.append({
                        "topic_id": tid,
                        "label": label,
                        "Positive": 0,
                        "Negative": 0,
                        "Neutral": 0,
                        "Total": 0
                    })

        topic_distribution = [
            {"topic_id": t["topic_id"], "label": t["label"], "count": t["post_count"], "percentage": t["percentage"]}
            for t in topics_info
        ]

        metadata = {
            "num_topics": chosen_k,
            "topics": topics_info,
            "top_keywords_overall": top_keywords_overall,
            "topic_distribution": topic_distribution,
            "topic_trends_over_time": topic_trends_over_time,
            "sentiment_by_topic": sentiment_by_topic
        }

        return df, metadata
