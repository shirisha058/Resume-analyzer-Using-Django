"""
ML Engine - AI Resume Analyzer
Primary model  : RoBERTa (stsb-roberta-base) via sentence-transformers
Fallback model : TF-IDF cosine similarity (used if RoBERTa not available)
Recommendations: Groq API — Llama 3.3 70B (rule-based fallback if API unavailable)

HOW SIMILARITY WORKS:
  1. RoBERTa encodes resume + JD into 768-dimensional vectors
  2. Cosine similarity is measured between the two vectors → base score
  3. Skill keyword overlap adds 40% weight to the final score
  4. Final score = (RoBERTa similarity × 0.6) + (skill overlap × 0.4)
"""

import re
import math
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
#  API KEY — reads from environment first, falls back to hardcoded value.
#  Best practice: set  GROQ_API_KEY=gsk_...  in your shell or .env file.
#  Get a free key at: https://console.groq.com
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    GROQ_API_KEY = None


# ─────────────────────────────────────────────────────────────────────────────
#  SKILL KEYWORD BANK
#  Each entry: (normalised_key, display_name)
#  normalised_key  — used for matching (always lowercase, no special chars)
#  display_name    — shown in the UI
#
#  FIX: skill comparison now uses normalised_key on BOTH sides, so
#       "Scikit-Learn" in resume and "scikit-learn" in JD both map to
#       "scikit-learn" → they match correctly.
# ─────────────────────────────────────────────────────────────────────────────
_RAW_SKILLS = [
    # ── Languages ────────────────────────────────────────────────────────────
    ("python",              "Python"),
    ("java",                "Java"),
    ("javascript",          "JavaScript"),
    ("typescript",          "TypeScript"),
    ("c++",                 "C++"),
    ("c#",                  "C#"),
    ("ruby",                "Ruby"),
    ("php",                 "PHP"),
    ("swift",               "Swift"),
    ("kotlin",              "Kotlin"),
    ("golang",              "Go"),
    ("rust",                "Rust"),
    ("scala",               "Scala"),
    ("matlab",              "MATLAB"),
    ("perl",                "Perl"),
    ("bash",                "Bash"),
    ("shell",               "Shell"),
    ("r",                   "R"),

    # ── Web / Frontend ───────────────────────────────────────────────────────
    ("html",                "HTML"),
    ("css",                 "CSS"),
    ("react",               "React"),
    ("angular",             "Angular"),
    ("vue",                 "Vue"),
    ("nextjs",              "Next.js"),
    ("nuxtjs",              "Nuxt.js"),
    ("tailwind",            "Tailwind"),
    ("bootstrap",           "Bootstrap"),
    ("figma",               "Figma"),
    ("photoshop",           "Photoshop"),

    # ── Backend / Frameworks ─────────────────────────────────────────────────
    ("django",              "Django"),
    ("flask",               "Flask"),
    ("fastapi",             "FastAPI"),
    ("nodejs",              "Node.js"),
    ("node.js",             "Node.js"),
    ("express",             "Express"),
    ("spring",              "Spring"),
    ("spring boot",         "Spring Boot"),
    ("laravel",             "Laravel"),
    ("rails",               "Rails"),

    # ── ML / AI / Data Science ───────────────────────────────────────────────
    ("machine learning",    "Machine Learning"),
    ("deep learning",       "Deep Learning"),
    ("nlp",                 "NLP"),
    ("natural language processing", "NLP"),
    ("computer vision",     "Computer Vision"),
    ("tensorflow",          "TensorFlow"),
    ("pytorch",             "PyTorch"),
    ("keras",               "Keras"),
    ("scikit-learn",        "Scikit-Learn"),
    ("scikit learn",        "Scikit-Learn"),   # ← alias: space variant
    ("sklearn",             "Scikit-Learn"),   # ← alias: short form
    ("pandas",              "Pandas"),
    ("numpy",               "NumPy"),
    ("matplotlib",          "Matplotlib"),
    ("seaborn",             "Seaborn"),
    ("data analysis",       "Data Analysis"),
    ("data science",        "Data Science"),
    ("statistics",          "Statistics"),
    ("regression",          "Regression"),
    ("classification",      "Classification"),
    ("clustering",          "Clustering"),
    ("neural network",      "Neural Network"),
    ("transformer",         "Transformer"),
    ("bert",                "BERT"),
    ("roberta",             "RoBERTa"),
    ("gpt",                 "GPT"),
    ("llm",                 "LLM"),
    ("apache spark",        "Apache Spark"),   # ← was missed before
    ("spark",               "Apache Spark"),
    ("etl",                 "ETL"),
    ("feature engineering", "Feature Engineering"),

    # ── Databases ────────────────────────────────────────────────────────────
    ("sql",                 "SQL"),
    ("mysql",               "MySQL"),
    ("postgresql",          "PostgreSQL"),
    ("mongodb",             "MongoDB"),
    ("redis",               "Redis"),
    ("elasticsearch",       "Elasticsearch"),
    ("sqlite",              "SQLite"),
    ("oracle",              "Oracle"),
    ("cassandra",           "Cassandra"),
    ("dynamodb",            "DynamoDB"),
    ("firebase",            "Firebase"),
    ("sap",                 "SAP"),            # ← added for finance roles

    # ── Cloud / DevOps ───────────────────────────────────────────────────────
    ("aws",                 "AWS"),
    ("azure",               "Azure"),
    ("gcp",                 "GCP"),
    ("docker",              "Docker"),
    ("kubernetes",          "Kubernetes"),
    ("terraform",           "Terraform"),
    ("ansible",             "Ansible"),
    ("jenkins",             "Jenkins"),
    ("ci/cd",               "CI/CD"),
    ("github actions",      "GitHub Actions"),
    ("git",                 "Git"),
    ("github",              "GitHub"),
    ("gitlab",              "GitLab"),
    ("linux",               "Linux"),
    ("nginx",               "Nginx"),
    ("apache",              "Apache"),
    ("prometheus",          "Prometheus"),     # ← added
    ("grafana",             "Grafana"),        # ← added

    # ── BI / Analytics ───────────────────────────────────────────────────────
    ("tableau",             "Tableau"),
    ("power bi",            "Power BI"),
    ("excel",               "Excel"),
    ("financial modeling",  "Financial Modeling"),  # ← added
    ("dcf",                 "DCF"),
    ("fp&a",                "FP&A"),

    # ── Testing / QA ─────────────────────────────────────────────────────────
    ("testing",             "Testing"),
    ("junit",               "JUnit"),
    ("selenium",            "Selenium"),
    ("testng",              "TestNG"),         # ← added
    ("cypress",             "Cypress"),        # ← added
    ("postman",             "Postman"),

    # ── Soft skills / Methodology ─────────────────────────────────────────────
    ("communication",       "Communication"),
    ("teamwork",            "Teamwork"),
    ("leadership",          "Leadership"),
    ("problem solving",     "Problem Solving"),
    ("agile",               "Agile"),
    ("scrum",               "Scrum"),
    ("project management",  "Project Management"),
    ("critical thinking",   "Critical Thinking"),
    ("time management",     "Time Management"),
    ("jira",                "JIRA"),

    # ── Other ─────────────────────────────────────────────────────────────────
    ("rest api",            "REST API"),
    ("graphql",             "GraphQL"),
    ("microservices",       "Microservices"),
    ("blockchain",          "Blockchain"),
    ("cybersecurity",       "Cybersecurity"),
]

# Build lookup: normalised_key → display_name  (deduplication: first wins)
_SKILL_MAP: dict[str, str] = {}
for _key, _display in _RAW_SKILLS:
    if _key not in _SKILL_MAP:
        _SKILL_MAP[_key] = _display

# Sorted longest-first so multi-word phrases match before their sub-words
_SKILL_KEYS_SORTED = sorted(_SKILL_MAP.keys(), key=len, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file):
    try:
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        pass
    return ""


def extract_text_from_docx(file):
    try:
        from docx import Document
        doc = Document(file)
        return " ".join(para.text for para in doc.paragraphs)
    except Exception:
        return ""


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        text = extract_text_from_pdf(uploaded_file)
    elif name.endswith('.docx'):
        text = extract_text_from_docx(uploaded_file)
    elif name.endswith('.txt'):
        text = uploaded_file.read().decode('utf-8', errors='ignore')
    else:
        text = ""
    return clean_text(text)


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\-\+\#\/&]', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  SKILL EXTRACTION
#  FIX: returns a list of normalised keys (lowercase).
#       The display name is looked up separately when needed.
#       This guarantees that matched/missing comparisons work correctly.
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(text: str) -> list[str]:
    """Return a list of normalised skill keys found in text."""
    text_lower = text.lower()
    found_keys: list[str] = []
    seen: set[str] = set()

    for key in _SKILL_KEYS_SORTED:
        if key in seen:
            continue
        # Use word-boundary regex; escape special chars in the key
        pattern = r'(?<![a-z0-9])' + re.escape(key) + r'(?![a-z0-9])'
        if re.search(pattern, text_lower):
            canonical = _SKILL_MAP[key]          # resolve alias to display name
            canonical_key = canonical.lower()    # use display name's lowercase as canonical
            if canonical_key not in seen:
                found_keys.append(canonical_key)
                seen.add(canonical_key)

    return found_keys


def display_skill(key: str) -> str:
    """Convert a normalised key back to its display name."""
    return _SKILL_MAP.get(key, key.title())


# ─────────────────────────────────────────────────────────────────────────────
#  SIMILARITY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    vocab = set(tokens1) | set(tokens2)

    def tfidf_vec(tokens):
        tf = Counter(tokens)
        total = len(tokens)
        return {w: tf[w] / total for w in vocab}

    v1 = tfidf_vec(tokens1)
    v2 = tfidf_vec(tokens2)

    dot  = sum(v1[w] * v2[w] for w in vocab)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def compute_roberta_similarity(text1: str, text2: str) -> float:
    """
    Semantic similarity using RoBERTa (stsb-roberta-base).
    Encodes both texts into 768-dimensional vectors and measures
    cosine similarity between them.
    Requires: pip install sentence-transformers torch
    Model (~500MB) is auto-downloaded on first run, then cached locally.
    """
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('stsb-roberta-base')
    emb1 = model.encode(text1[:512], convert_to_tensor=True)
    emb2 = model.encode(text2[:512], convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return max(0.0, min(1.0, score))


def compute_similarity(resume_text, jd_text):
    try:
        score = compute_roberta_similarity(resume_text, jd_text)
        print(f"[ML Engine] RoBERTa similarity: {round(score * 100, 1)}%")
        return score, "RoBERTa"
    except Exception as e:
        print(f"[ML Engine] RoBERTa failed ({e}) — falling back to TF-IDF")
        score = compute_tfidf_similarity(resume_text, jd_text)
        print(f"[ML Engine] TF-IDF similarity: {round(score * 100, 1)}%")
        return score, "TF-IDF"


# ─────────────────────────────────────────────────────────────────────────────
#  RECOMMENDATION — Groq API with rule-based fallback
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendation(score: float, matched_keys: list, missing_keys: list) -> str:
    score_pct   = round(score * 100, 1)
    matched_str = ", ".join(display_skill(s) for s in matched_keys[:10]) or "None"
    missing_str = ", ".join(display_skill(s) for s in missing_keys[:10]) or "None"

    if not GROQ_API_KEY:
        print("[ML Engine] Groq API key not set — using rule-based recommendation")
        return _rule_based_recommendation(score_pct, matched_keys, missing_keys)

    prompt = (
        f"You are an expert career counselor reviewing a resume-to-job match result.\n\n"
        f"Match score: {score_pct}%\n"
        f"Skills the candidate HAS that match the job: {matched_str}\n"
        f"Skills the job REQUIRES but the candidate is MISSING: {missing_str}\n\n"
        f"Write a helpful recommendation in 3-4 sentences.\n"
        f"- Mention the match score result clearly\n"
        f"- Name the missing skills specifically\n"
        f"- Give one actionable piece of advice\n"
        f"- Be encouraging but honest\n"
        f"- Do NOT use bullet points, just plain paragraph text"
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    body = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
    }).encode("utf-8")

    import requests as _requests
    for attempt in range(3):
        try:
            resp = _requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "User-Agent": "Mozilla/5.0",
                },
                data=body,
                timeout=15,
            )
            if resp.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"[ML Engine] Rate limited, waiting {wait}s and retrying...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            recommendation = data["choices"][0]["message"]["content"]
            print("[ML Engine] Groq API recommendation generated successfully")
            return recommendation.strip()
        except Exception as e:
            print(f"[ML Engine] Groq attempt {attempt+1} failed: {e}")
            if attempt == 2:
                break

    print("[ML Engine] Using rule-based fallback recommendation")
    return _rule_based_recommendation(score_pct, matched_keys, missing_keys)


def _rule_based_recommendation(score_pct: float, matched: list, missing: list) -> str:
    missing_display = ", ".join(display_skill(s) for s in missing[:5])

    if score_pct >= 80:
        msg = (
            f"Excellent match at {score_pct}%! Your profile aligns very strongly with this role "
            f"with {len(matched)} matching skills. "
        )
        if missing:
            msg += f"To make your application even stronger, consider adding expertise in: {missing_display}."
        else:
            msg += "You appear to meet all key requirements — apply with confidence!"

    elif score_pct >= 60:
        msg = (
            f"Good match at {score_pct}%. You meet many core requirements with {len(matched)} matching skills. "
        )
        if missing:
            msg += f"Bridging the gap in {missing_display} would significantly strengthen your candidacy."
        else:
            msg += "Highlight your relevant experience in your cover letter."

    elif score_pct >= 40:
        msg = (
            f"Moderate match at {score_pct}%. Your resume partially aligns with this role "
            f"({len(matched)} matching skills). "
        )
        if missing:
            msg += f"Priority areas to develop: {missing_display}. Consider online courses in these areas."

    else:
        msg = f"Low match at {score_pct}%. This role requires skills that differ from your current profile. "
        if missing:
            msg += f"Key skills to focus on: {missing_display}. "
        msg += "Consider roles that better match your existing skills, or invest in upskilling first."

    return msg


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_resume(resume_file, job_description: str, job_title: str = "") -> dict:
    print(f"\n[ML Engine] ── Starting analysis ──")
    print(f"[ML Engine] File: {resume_file.name}")

    resume_text = extract_text(resume_file)
    jd_text     = clean_text(job_description)

    if not resume_text:
        return {"error": "Could not extract text from resume. Please use PDF, DOCX, or TXT format."}

    print(f"[ML Engine] Resume text length: {len(resume_text)} chars")
    print(f"[ML Engine] JD text length:     {len(jd_text)} chars")

    similarity_score, method = compute_similarity(resume_text, jd_text)

    # FIX: both sides return normalised keys → comparison is always apples-to-apples
    resume_skill_keys = extract_skills(resume_text)
    jd_skill_keys     = extract_skills(jd_text)

    print(f"[ML Engine] Resume skills: {[display_skill(k) for k in resume_skill_keys]}")
    print(f"[ML Engine] JD skills:     {[display_skill(k) for k in jd_skill_keys]}")

    resume_set = set(resume_skill_keys)
    jd_set     = set(jd_skill_keys)

    matched_keys = [k for k in resume_skill_keys if k in jd_set]
    missing_keys = [k for k in jd_skill_keys     if k not in resume_set]

    print(f"[ML Engine] Matched: {[display_skill(k) for k in matched_keys]}")
    print(f"[ML Engine] Missing: {[display_skill(k) for k in missing_keys]}")

    if jd_skill_keys:
        skill_overlap = len(matched_keys) / len(jd_skill_keys)
        final_score   = (similarity_score * 0.6) + (skill_overlap * 0.4)
    else:
        final_score = similarity_score

    print(f"[ML Engine] Final score: {round(final_score * 100, 1)}%")

    recommendation = generate_recommendation(final_score, matched_keys, missing_keys)

    print(f"[ML Engine] ── Analysis complete ──\n")

    # Convert keys back to display names for the view layer
    matched_display = [display_skill(k) for k in matched_keys]
    missing_display = [display_skill(k) for k in missing_keys]

    return {
        "resume_text"        : resume_text[:3000],
        "match_score"        : round(final_score * 100, 1),
        "matched_skills"     : matched_display,
        "missing_skills"     : missing_display,
        "recommendation"     : recommendation,
        "method"             : method,
        "resume_skills_count": len(resume_skill_keys),
        "jd_skills_count"    : len(jd_skill_keys),
    }
