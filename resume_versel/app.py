import os
import re
import base64
import json
from http.server import BaseHTTPRequestHandler
from io import BytesIO

# --- Dependencies to install (from requirements.txt) ---
import fitz  # PyMuPDF
from docx import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# ALL LOGIC FROM core.py and llm_manager.py IS NOW IN THIS FILE
# ==============================================================================

# --- Logic from llm_manager.py ---
llm = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[llm_manager] Failed to initialize LLM: {e}")
else:
    print("[llm_manager] GOOGLE_API_KEY not configured.")

def get_semantic_match_score(resume_text: str, jd_text: str) -> float:
    """Return a float score from 0 to 50."""
    if llm is None:
        return 0.0
    try:
        prompt_template = PromptTemplate.from_template(
            """You are an assistant to evaluate how well a candidate's resume matches a job description.
Provide a relevance score (integer) from 0 to 100, where 100 means perfect match.
**Respond ONLY with one integer.**

Resume:
{resume}

Job Description:
{jd}
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.invoke({"resume": resume_text, "jd": jd_text})
        text = response.get("text", "").strip()
        m = re.search(r'\d+', text)
        if m:
            val = int(m.group())
            val = max(0, min(val, 100)) # Clamp between 0 and 100
            return val * 0.5
        return 25.0 # Fallback score
    except Exception as e:
        print(f"[llm_manager] Error in get_semantic_match_score: {e}")
        return 0.0

def get_feedback_and_suggestions(resume_text: str, jd_text: str) -> str:
    if llm is None:
        return "LLM not available. Please set the GOOGLE_API_KEY."
    try:
        prompt_template = PromptTemplate.from_template(
            """You are a resume improvement advisor. Compare the candidate’s resume to the job description.
Provide **bullet point suggestions** to help the candidate improve the resume, focusing on:
- missing technical AND soft skills
- relevant project or experience highlighting
- using terminology from the JD
- formatting and clarity if needed

Respond with bullet points only.
Job Description:
{jd}

Resume:
{resume}

Suggestions:
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.invoke({"resume": resume_text, "jd": jd_text})
        text = response.get("text", "").strip()
        return text
    except Exception as e:
        print(f"[llm_manager] Error in get_feedback_and_suggestions: {e}")
        return "Could not generate suggestions due to an error."

# --- Logic from core.py ---
STOP_WORDS = {
    'and', 'the', 'of', 'in', 'to', 'a', 'with', 'for', 'on',
    'is', 'are', 'that', 'by', 'as', 'this', 'an', 'or', 'at',
    'from', 'it', 'be', 'which', 'you', 'we'
}
MAX_RESUME_SIZE_MB = 200
KNOWN_SKILLS = {
    'python', 'java', 'sql', 'excel', 'machine learning', 'deep learning',
    'communication', 'teamwork', 'project management', 'docker', 'aws',
    'javascript', 'react', 'nodejs', 'git', 'linux'
}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"[core] Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = Document(BytesIO(file_bytes))
        text = "\n".join(para.text for para in doc.paragraphs)
        return text
    except Exception as e:
        print(f"[core] Error reading DOCX: {e}")
        return ""

def get_hard_match_score(resume_text: str, jd_text: str):
    jd_words_all = set(re.findall(r'\b\w+\b', jd_text.lower()))
    resume_words_all = set(re.findall(r'\b\w+\b', resume_text.lower()))
    jd_skills = {w for w in jd_words_all if w not in STOP_WORDS and w in KNOWN_SKILLS}
    resume_skills = {w for w in resume_words_all if w not in STOP_WORDS and w in KNOWN_SKILLS}
    if not jd_skills:
        return 0.0, [], []
    matched = resume_skills.intersection(jd_skills)
    missing = jd_skills.difference(resume_skills)
    score = (len(matched) / len(jd_skills)) * 50
    return score, sorted(list(matched)), sorted(list(missing))

def is_too_large(file_bytes: bytes) -> bool:
    return (len(file_bytes) / (1024 * 1024)) > MAX_RESUME_SIZE_MB

def format_skills_list(skills_list):
    return ", ".join(skills_list) if skills_list else "None"

# ==============================================================================
# MAIN HANDLER FOR VERCEL (REPLACES THE FLASK APP)
# ==============================================================================
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data_bytes = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data_bytes.decode('utf-8'))
            job_description_data = body.get('job_description', '')
            resumes_data = body.get('resumes', [])

            # This is the main logic from your original app.py/analyze_resumes route
            job_description_text = ""
            if isinstance(job_description_data, str) and job_description_data.startswith("data:"):
                header, jd_base64 = job_description_data.split(',', 1)
                jd_bytes = base64.b64decode(jd_base64)
                if 'pdf' in header.lower():
                    job_description_text = extract_text_from_pdf(jd_bytes)
                elif 'wordprocessingml' in header.lower():
                    job_description_text = extract_text_from_docx(jd_bytes)
            else:
                job_description_text = job_description_data

            results = []
            for res in resumes_data:
                file_name, content = res.get('fileName'), res.get('content')
                header, base64_content = content.split(',', 1)
                file_bytes = base64.b64decode(base64_content)
                
                resume_text = ""
                if file_name.lower().endswith('.pdf'): resume_text = extract_text_from_pdf(file_bytes)
                elif file_name.lower().endswith('.docx'): resume_text = extract_text_from_docx(file_bytes)

                candidate_name = next((line.strip() for line in resume_text.strip().splitlines()[:5] if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line.strip())), "N/A")
                email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', resume_text, re.IGNORECASE)
                candidate_email = email_match.group(0) if email_match else "N/A"

                hard_score, matched_skills, missing_skills = get_hard_match_score(resume_text, job_description_text)
                semantic_score = get_semantic_match_score(resume_text, job_description_text)
                suggestions = get_feedback_and_suggestions(resume_text, job_description_text)
                total_score = round(hard_score + semantic_score)
                verdict = "High" if total_score >= 80 else "Medium" if total_score >= 50 else "Low"
                
                results.append({
                    "resumeName": file_name, "candidateName": candidate_name, "candidateEmail": candidate_email,
                    "score": total_score, "verdict": verdict, "matchedSkills": matched_skills,
                    "matchedSkillsFormatted": format_skills_list(matched_skills), "missingSkills": missing_skills,
                    "missingSkillsFormatted": format_skills_list(missing_skills), "suggestions": suggestions
                })
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(results).encode('utf-8'))

        except Exception as e:
            # Send error response
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e), "type": type(e).__name__}).encode('utf-8'))
