import os
import re
import base64
import json
from http.server import BaseHTTPRequestHandler
from io import BytesIO

# --- Dependencies ---
import fitz  # PyMuPDF
from docx import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# CONSOLIDATED LOGIC
# ==============================================================================

# --- Initialize LLM ---
llm = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[api] Failed to initialize LLM: {e}")

# --- Combined LLM analysis function ---
def get_llm_analysis(resume_text: str, jd_text: str) -> dict:
    if not llm:
        return {"score": 0, "suggestions": "LLM not available. Check API key."}
    try:
        prompt = PromptTemplate.from_template(
            """Analyze the resume against the job description. Respond ONLY with a single, valid JSON object with two keys: "score" and "suggestions".
- "score": An integer (0-100) for match quality.
- "suggestions": A brief string of actionable resume improvement advice, formatted with bullet points (e.g., "- suggestion one\\n- suggestion two").

Example of a valid response:
{{"score": 75, "suggestions": "- Highlight cloud experience like AWS.\\n- Quantify achievements in past projects with metrics."}}

JD: {jd}
Resume: {resume}
JSON Response:
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response_text = chain.invoke({"resume": resume_text, "jd": jd_text}).get("text", "{}")
        
        # More robust JSON parsing
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            print(f"[api] AI response did not contain JSON: {response_text}")
            return {"score": 0, "suggestions": "AI response format error."}

        clean_json_str = json_match.group(0)
        parsed_json = json.loads(clean_json_str)

        score_val = parsed_json.get("score", 0)
        suggestions_val = parsed_json.get("suggestions", "No suggestions generated.")
        
        # Normalize score to be out of 50 for the semantic portion of the total score
        normalized_score = (int(score_val) / 100) * 50
        
        return {"score": max(0, min(normalized_score, 50)), "suggestions": suggestions_val}
    except Exception as e:
        print(f"[api] Critical error in get_llm_analysis: {e}")
        return {"score": 0, "suggestions": "Error during AI analysis. Check Vercel logs."}

# --- Text extraction and keyword matching logic ---
STOP_WORDS = {'and', 'the', 'of', 'in', 'to', 'a', 'with', 'for', 'on', 'is', 'are'}
KNOWN_SKILLS = {'python', 'java', 'sql', 'aws', 'docker', 'react', 'javascript', 'teamwork'}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"[api] Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        return "\n".join(para.text for para in Document(BytesIO(file_bytes)).paragraphs)
    except Exception as e:
        print(f"[api] Error reading DOCX: {e}")
        return ""

def get_hard_match_score(resume_text: str, jd_text: str):
    jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    jd_skills = {w for w in jd_words if w not in STOP_WORDS and w in KNOWN_SKILLS}
    if not jd_skills:
        # If no relevant skills in JD, return 0 score but show what skills the resume DOES have
        resume_known_skills = KNOWN_SKILLS.intersection(resume_words)
        return 0.0, [], list(resume_known_skills)
    resume_skills = KNOWN_SKILLS.intersection(resume_words)
    matched = resume_skills.intersection(jd_skills)
    missing = jd_skills.difference(resume_skills)
    score = (len(matched) / len(jd_skills)) * 50 if jd_skills else 0
    return score, sorted(list(matched)), sorted(list(missing))

# ==============================================================================
# MAIN HANDLER FOR VERCEL
# ==============================================================================
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data_bytes = self.rfile.read(content_length)
            body = json.loads(post_data_bytes.decode('utf-8'))
            
            job_description_data = body.get('job_description', '')
            resumes_data = body.get('resumes', [])

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

                if not resume_text: continue

                email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', resume_text, re.IGNORECASE)
                candidate_email = email_match.group(0) if email_match else "N/A"

                hard_score, matched_skills, missing_skills = get_hard_match_score(resume_text, job_description_text)
                llm_results = get_llm_analysis(resume_text, job_description_text)
                
                total_score = round(hard_score + llm_results["score"])
                verdict = "High" if total_score >= 80 else "Medium" if total_score >= 50 else "Low"
                
                results.append({
                    "resumeName": file_name,
                    "candidateEmail": candidate_email,
                    "score": total_score,
                    "verdict": verdict,
                    "missingSkills": missing_skills,
                    "suggestions": llm_results["suggestions"]
                })
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(results).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"An internal server error occurred: {str(e)}"}).encode('utf-8'))
