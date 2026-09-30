# AI-Powered Social Media Sentiment & Trend Analysis Dashboard

An enterprise NLP platform for social media intelligence featuring real BERT sentiment classification, unsupervised LDA topic clustering, hashtag analytics, and Microsoft Power BI export.

---

## 🚀 Quick Start (Currently Running!)

Both the backend and frontend are **already running** in your environment:

- **Frontend UI**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000) (Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs))

Simply open [http://localhost:5173](http://localhost:5173) in your browser!

---

## 🛠️ How to Run From Scratch

If you ever need to restart or run the project manually, follow these two steps:

### Terminal 1: Start the Backend (FastAPI + PyTorch)

```powershell
# Navigate to project root
cd d:\MLops

# Run the FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

*The backend will initialize the CardiffNLP Twitter RoBERTa model and listen on `http://127.0.0.1:8000`.*

---

### Terminal 2: Start the Frontend (React + Vite)

```powershell
# Navigate to frontend folder
cd d:\MLops\frontend

# Start Vite dev server
npm run dev
```

*Open your browser and navigate to `http://localhost:5173`.*

---

## 📂 Expected Input Dataset Structure

The application supports **CSV**, **Excel (.xlsx)**, and **JSON** files.

| Column | Required? | Variations Auto-Detected | Description |
| :--- | :---: | :--- | :--- |
| **`text`** | **YES** | `text`, `tweet`, `content`, `post`, `comment`, `message`, `body` | The social media post text to analyze |
| **`post_id`** | Optional | `post_id`, `id`, `tweet_id`, `comment_id`, `uid` | Unique identifier (auto-generated if missing) |
| **`date`** | Optional | `date`, `created_at`, `timestamp`, `datetime`, `time` | Enables temporal trend charts |
| **`platform`** | Optional | `platform`, `source`, `channel`, `network` | Enables cross-platform sentiment comparison |
| **`hashtags`** | Optional | `hashtags`, `tags`, `hashtag`, `tag` | Enables hashtag analytics (also auto-extracted from text) |
| **`sentiment`** | Optional | `sentiment`, `label`, `target`, `ground_truth` | If present, calculates Accuracy, Precision, Recall, F1 |

---

## 📊 End-to-End Workflow

1. **Upload Dataset**: Drag & drop or select your CSV, Excel, or JSON file.
2. **Review Schema**: Confirm the detected columns (or pick them from dropdowns).
3. **Configure**: Select number of LDA topic clusters (e.g. 5) and toggle deduplication.
4. **Run Analysis**: Click **Start BERT & LDA Analysis**.
5. **Explore Analytics**:
   - **Overview Dashboard**: KPI cards, Sentiment Donut, Confidence Histogram, Sentiment Over Time, Platform Breakdown, and classified posts table.
   - **Topic Modeling (LDA)**: Topic cards with keyword weights, topic volume distribution, and sentiment by topic.
   - **Hashtag Trends**: Top hashtag frequencies, sentiment per tag, and temporal curves.
6. **Power BI Export**:
   - Go to the **Power BI Export** tab.
   - Click **Download CSV** or **Download Excel**.
   - Copy pre-built DAX measures into Power BI Desktop for instant executive reporting.
