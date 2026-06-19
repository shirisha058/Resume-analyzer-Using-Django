# ResumeIQ — AI Resume Analyzer

An AI-powered resume analyzer built with Django that matches resumes to job descriptions using TF-IDF similarity and skill extraction, with AI-generated recommendations powered by the Groq API (Llama 3.3).

---

## Features

- Upload resume (PDF, DOCX, TXT)
- Paste job description
- Get match score (0–100%)
- See matched skills and missing skills
- Get AI-generated personalized recommendation (via Groq API)
- Rule-based recommendation fallback (if no API key)
- View analysis history

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 5.1 |
| Similarity | TF-IDF cosine similarity |
| Skill Extraction | Custom keyword bank (100+ skills) |
| AI Recommendations | Groq API — Llama 3.3 70B |
| PDF parsing | pdfplumber |
| DOCX parsing | python-docx |
| HTTP client | requests |
| Database | SQLite |
| Frontend | HTML, CSS, Vanilla JS |

---

## Project Structure

```
AI-Resume_Analyzer_2/
├── manage.py
├── requirements.txt
├── db.sqlite3
├── resume_analyzer/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── analyzer/
│   ├── models.py          ← AnalysisResult DB model
│   ├── views.py           ← Django views
│   ├── urls.py            ← URL routing
│   ├── ml_engine.py       ← TF-IDF + Groq AI pipeline
│   └── templates/
│       └── analyzer/
│           ├── base.html      ← Navbar, footer, global styles
│           ├── home.html      ← Upload form page
│           ├── result.html    ← Analysis results page
│           └── history.html   ← Past analyses
├── static/
└── media/
```

---

## Setup & Run

### Step 1 — Get a free Groq API key (required for AI recommendations)

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free — no credit card needed)
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_` — shown only once)

### Step 2 — Clone and create virtual environment

```bash
git clone <your-repo-url>
cd AI-Resume_Analyzer_2
python -m venv demo
demo\Scripts\activate        # Windows
# source demo/bin/activate   # Mac/Linux
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add your Groq API key

Open `analyzer/ml_engine.py` and find line ~20:

```python
# ← Paste your Groq API key here
# Get a free key at: https://console.groq.com
_FALLBACK_KEY = ""
```

Replace the empty string with your key:

```python
_FALLBACK_KEY = "gsk_YourKeyHere"
```

### Step 5 — Run migrations

```bash
python manage.py migrate
```

### Step 6 — Start server

```bash
python manage.py runserver
```

### Step 7 — Open browser

```
http://127.0.0.1:8000
```

---

## Every Time You Run (after first setup)

```bash
cd "AI-Resume_Analyzer_2"
demo\Scripts\activate
python manage.py runserver
```

---

## How It Works

```
User uploads resume (PDF / DOCX / TXT)
             ↓
Text extraction (pdfplumber / python-docx)
             ↓
Clean and normalize text
             ↓
TF-IDF cosine similarity → base score
             ↓
Skill keyword matching (100+ skills) → skill overlap score
             ↓
Final score = (TF-IDF × 0.6) + (skill overlap × 0.4)
             ↓
Groq API (Llama 3.3) → AI recommendation
             ↓
Rule-based fallback if API unavailable
             ↓
Django saves result → shows results page
```

---

## ML Pipeline (ml_engine.py)

| Function | Description |
|---|---|
| `extract_text()` | Reads PDF / DOCX / TXT and returns clean plain text |
| `extract_skills()` | Scans text for 100+ skill keywords (normalised, alias-aware) |
| `compute_tfidf_similarity()` | TF-IDF cosine similarity between resume and JD |
| `generate_recommendation()` | Calls Groq API for AI recommendation, falls back to rule-based |
| `analyze_resume()` | Main pipeline that calls all of the above |

---

## Groq API Notes

- **Free tier limits:** 30 requests/minute, 500,000 tokens/day
- **Model used:** `llama-3.3-70b-versatile`
- **Key never expires** unless you delete it from console.groq.com
- **If you see 403 in logs:** your key is invalid — get a new one from console.groq.com and update `_FALLBACK_KEY` in `ml_engine.py`
- **Fallback:** if the API fails for any reason, a rule-based recommendation is used automatically — the app never crashes



## Troubleshooting

| Problem | Fix |
|---|---|
| `Groq API key is invalid (403)` | Get new key at console.groq.com, update `_FALLBACK_KEY` |
| `model_decommissioned` error | Change model to `llama-3.3-70b-versatile` in `ml_engine.py` |
| Low match score | Check resume has enough content (must be 1000+ chars) |
| Skills not detected | Ensure skills are written in full (e.g. "Python" not just "Py") |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in activated venv |

---

## Future Improvements

- Add user authentication
- Add resume improvement suggestions
- Add ATS score calculation
- Upgrade to RoBERTa semantic similarity
- Deploy to Render / Railway


## work flow
User uploads Resume (PDF/DOCX/TXT)
            +  pastes Job Description
                     ↓
         ┌─── Text Extraction ───┐
         │  pdfplumber / docx    │
         └───────────────────────┘
                     ↓
         ┌─── Skill Extraction ──┐
         │  100+ keyword bank    │
         │  resume skills        │
         │  JD skills            │
         └───────────────────────┘
                     ↓
         ┌─── Similarity Score ──┐
         │  RoBERTa              │
         │                       │
         │  → 0 to 1 score       │
         └───────────────────────┘
                     ↓
         ┌─── Final Score ───────┐
         │  60% similarity       │
         │  40% skill overlap    │
         │  → 0 to 100%          │
         └───────────────────────┘
                     ↓
         ┌─── AI Recommendation ─┐
         │  Groq API             │
         │  Llama 3.3 70B        │
         └───────────────────────┘
                     ↓
         ┌─── Django saves ──────┐
         │  SQLite database      │
         │  Shows result page    │
         └───────────────────────┘