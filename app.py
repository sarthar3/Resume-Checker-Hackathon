import streamlit as st
import pdfplumber
import docx2txt
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import openai
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True
if not check_password():
    st.stop()

st.set_page_config(page_title="Resume Relevance Checker", layout="wide")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please install spaCy model: python -m spacy download en_core_web_sm")
    st.stop()

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + " "
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()

def extract_skills(text):
    doc = nlp(text)
    skills = set()
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECH"]:
            skills.add(ent.text.lower())
    
    skill_patterns = [
        r"python", r"sql", r"mysql", r"postgresql", r"power bi", r"tableau", 
        r"pandas", r"numpy", r"matplotlib", r"seaborn", r"scikit-learn", 
        r"tensorflow", r"pytorch", r"spark", r"kafka", r"excel", r"dax", 
        r"power query", r"java", r"c\+\+", r"javascript", r"html", r"css", 
        r"react", r"node", r"django", r"flask", r"aws", r"azure", r"gcp", 
        r"docker", r"kubernetes", r"machine learning", r"deep learning", 
        r"nlp", r"computer vision", r"data analysis", r"data visualization",
        r"web scraping", r"beautiful soup", r"requests", r"statistics",
        r"exploratory data analysis", r"eda", r"data cleaning", r"git"
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.update([match.lower() for match in matches])
    
    return list(skills)

def extract_dynamic_skills(text, top_n=20):
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    words = text_clean.lower().split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    important_skills = [
        "python", "sql", "machine", "learning", "data", "analysis", 
        "power", "bi", "tableau", "excel", "pandas", "numpy",
        "visualization", "statistics", "spark", "kafka", "aws",
        "azure", "database", "mysql", "postgresql", "analysis",
        "processing", "engineer", "developer", "analyst"
    ]
    
    extracted = [word for word, freq in sorted_words[:top_n] if word in important_skills]
    
    for skill in important_skills:
        if skill in text.lower() and skill not in extracted:
            extracted.append(skill)
    
    return extracted[:top_n]

def extract_projects_and_certs(text):
    text_lower = text.lower()
    
    project_patterns = [
        r"project.*?:.*?([^\n]+)", 
        r"developed.*?system", 
        r"built.*?application",
        r"created.*?dashboard",
        r"designed.*?database",
        r"implemented.*?model"
    ]
    
    projects = []
    for pattern in project_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        projects.extend(matches)
    
    cert_patterns = [
        r"certification.*?:.*?([^\n]+)",
        r"certified.*?([^\n]+)",
        r"coursera", 
        r"udemy", 
        r"edx",
        r"nanodegree",
        r"ibm.*?certificate",
        r"microsoft.*?certificate",
        r"google.*?certificate"
    ]
    
    certs = []
    for pattern in cert_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        certs.extend(matches)
    
    return list(set(projects))[:5], list(set(certs))[:5]

def compute_semantic_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0] * 100

def compute_score_with_projects(resume_text, jd_text, jd_keywords):
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    matches = sum(1 for kw in jd_keywords if kw.lower() in resume_lower)
    hard_score = (matches / len(jd_keywords)) * 100 if jd_keywords else 0
    
    sim = compute_semantic_similarity(resume_text, jd_text)
    
    final = 0.7 * hard_score + 0.3 * sim
    
    if final >= 75:
        verdict = "High"
    elif final >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"
    
    projects, certs = extract_projects_and_certs(resume_text)
    
    return {
        "final_score": round(final, 2),
        "verdict": verdict,
        "keyword_match": round(hard_score, 2),
        "semantic_match": round(sim, 2),
        "projects": ", ".join(projects) if projects else "None",
        "certifications": ", ".join(certs) if certs else "None"
    }

def generate_feedback(resume_text, jd_text, missing_skills, matched_skills):
    if not missing_skills:
        return "Resume has all required skills. Good match for the position!"
    
    truncated_resume = resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text
    truncated_jd = jd_text[:2000] + "..." if len(jd_text) > 2000 else jd_text
    
    prompt = f"""
    As a career advisor, analyze this resume against the job description and provide constructive feedback.
    
    JOB DESCRIPTION:
    {truncated_jd}
    
    RESUME:
    {truncated_resume}
    
    The resume is missing these key skills: {', '.join(missing_skills)}
    The resume has these matching skills: {', '.join(matched_skills[:10])}
    
    Provide specific, actionable suggestions to improve the resume for this job. Focus on:
    1. How to highlight existing skills better
    2. What specific skills to acquire or emphasize
    3. Any formatting or content improvements
    Keep it concise (3-4 sentences maximum).
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful career advisor providing specific, actionable resume feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Focus on acquiring these skills: {', '.join(missing_skills[:5])}. Error: {str(e)}"

def init_db():
    conn = sqlite3.connect("resume_results.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume TEXT,
        final_score REAL,
        verdict TEXT,
        keyword_match REAL,
        semantic_match REAL,
        missing_skills TEXT,
        projects TEXT,
        certifications TEXT,
        feedback TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

def save_to_db(results):
    conn = sqlite3.connect("resume_results.db")
    c = conn.cursor()
    for r in results:
        c.execute("""
        INSERT INTO results (resume, final_score, verdict, keyword_match, semantic_match, 
                           missing_skills, projects, certifications, feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (r["Resume"], r["Final Score"], r["Verdict"], r["Keyword Match %"],
              r["Semantic Match %"], r["Missing Skills"], r["Projects"], 
              r["Certifications"], r["Feedback"]))
    conn.commit()
    conn.close()

st.title("📄 Automated Resume Relevance Checker")
st.markdown("Upload JD + Resumes to evaluate relevance, show missing skills, feedback, and visualizations.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Enter your OpenAI API key to enable feedback generation")
    if api_key:
        openai.api_key = api_key
        st.success("API key set successfully!")
    else:
        st.warning("Please enter your OpenAI API key to enable feedback generation")

jd_file = st.file_uploader("📌 Upload Job Description (txt/docx/pdf)", type=["txt","docx","pdf"])
resumes = st.file_uploader("📌 Upload Resumes (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True)

if jd_file and resumes:
    jd_text = extract_text(jd_file)
    jd_keywords = extract_dynamic_skills(jd_text)
    
    st.subheader("📋 JD Extracted Keywords")
    st.write(", ".join(jd_keywords))

    results = []
    for res in resumes:
        resume_text = extract_text(res)
        resume_skills = extract_skills(resume_text)
        score_data = compute_score_with_projects(resume_text, jd_text, jd_keywords)
        
        missing = [skill for skill in jd_keywords if skill not in resume_skills]
        matched = [skill for skill in jd_keywords if skill in resume_skills]
        
        feedback = "API key not provided" if not api_key else generate_feedback(resume_text, jd_text, missing, matched)

        results.append({
            "Resume": res.name,
            "Final Score": score_data["final_score"],
            "Verdict": score_data["verdict"],
            "Keyword Match %": score_data["keyword_match"],
            "Semantic Match %": score_data["semantic_match"],
            "Missing Skills": ", ".join(missing) if missing else "None",
            "Projects": score_data["projects"],
            "Certifications": score_data["certifications"],
            "Feedback": feedback
        })

    save_to_db(results)

    st.subheader("📊 Results Dashboard")
    df = pd.DataFrame(results)

    verdict_filter = st.multiselect("Filter by Verdict", options=["High","Medium","Low"], default=["High","Medium","Low"])
    score_filter = st.slider("Minimum Final Score", 0, 100, 0)

    df_filtered = df[(df["Verdict"].isin(verdict_filter)) & (df["Final Score"] >= score_filter)]
    df_filtered = df_filtered.sort_values(by="Final Score", ascending=False)
    
    df_filtered["Matched Skills Count"] = df_filtered.apply(
        lambda x: len(jd_keywords) - (0 if x["Missing Skills"]=="None" else len(x["Missing Skills"].split(','))), 
        axis=1
    )
    df_filtered["Missing Skills Count"] = df_filtered["Missing Skills"].apply(
        lambda x: 0 if x=="None" else len(x.split(','))
    )
    
    st.dataframe(df_filtered, use_container_width=True)

    st.download_button("⬇️ Download Results (CSV)", df_filtered.to_csv(index=False).encode("utf-8"), "resume_results.csv","text/csv")

    st.subheader("📈 Resume Scores Visualization")
    if not df_filtered.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = []
        for score in df_filtered["Final Score"]:
            if score >= 75:
                colors.append('green')
            elif score >= 50:
                colors.append('orange')
            else:
                colors.append('red')
        ax.barh(df_filtered["Resume"], df_filtered["Final Score"], color=colors)
        ax.set_xlabel("Final Score")
        ax.set_ylabel("Resumes")
        ax.set_xlim(0, 100)
        for i, v in enumerate(df_filtered["Final Score"]):
            ax.text(v + 1, i, str(v), va='center')
        st.pyplot(fig)
    
    st.subheader("🟢 Matched vs 🔴 Missing Skills per Resume")
    if not df_filtered.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, len(df_filtered)*0.5)))
        ax.barh(df_filtered["Resume"], df_filtered["Matched Skills Count"], color='green', label='Matched Skills')
        ax.barh(df_filtered["Resume"], df_filtered["Missing Skills Count"], left=df_filtered["Matched Skills Count"], color='red', label='Missing Skills')
        ax.set_xlabel("Number of Skills")
        ax.set_ylabel("Resumes")
        ax.legend()
        st.pyplot(fig)
        st.success("Results saved to database successfully!")

    st.subheader("🏆 Projects & Certifications Overview")
    project_counts = df_filtered["Projects"].apply(lambda x: 0 if x == "None" else len(x.split(','))).sum()
    cert_counts = df_filtered["Certifications"].apply(lambda x: 0 if x == "None" else len(x.split(','))).sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Projects Found", project_counts)
    with col2:
        st.metric("Total Certifications Found", cert_counts)
    
    for _, row in df_filtered.iterrows():
        with st.expander(f"📋 {row['Resume']} - Projects & Certifications"):
            st.write(f"**Projects:** {row['Projects']}")
            st.write(f"**Certifications:** {row['Certifications']}")
