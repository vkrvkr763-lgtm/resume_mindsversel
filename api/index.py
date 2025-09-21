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

# --- NEW: Single, intelligent AI analysis function ---
def get_ai_analysis(resume_text: str, jd_text: str) -> dict:
    if not llm:
        return {"score": 0, "suggestions": "LLM not available.", "missing_skills": []}
    try:
        # This new prompt asks the AI to do all the work in one go.
        prompt = PromptTemplate.from_template(
            """CRITICAL: Your response MUST be a single, valid JSON object and nothing else.
The JSON object must contain three keys: "score", "missing_skills", and "suggestions".
- "score": An integer (0-100) representing the overall match quality.
- "missing_skills": An array of strings listing the key skills mentioned in the job description that are missing from the resume.
- "suggestions": A brief string of actionable resume improvement advice, using newline characters (\\n) for bullet points.

Example of a perfect response:
{{"score": 65, "missing_skills": ["Excel", "Tableau"], "suggestions": "- Add a section for software skills like Excel.\\n- Mention any data visualization projects."}}

Job Description:
{jd}

Resume:
{resume}
JSON Response:
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response_text = chain.invoke({"resume": resume_text, "jd": jd_text}).get("text", "{}")
        
        # Aggressive JSON cleaning
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("AI response did not contain valid JSON.")

        clean_json_str = json_match.group(0)
        parsed_json = json.loads(clean_json_str)

        return {
            "score": parsed_json.get("score", 0),
            "missing_skills": parsed_json.get("missing_skills", []),
            "suggestions": parsed_json.get("suggestions", "No suggestions generated.")
        }
    except Exception as e:
        print(f"[api] Critical error in get_ai_analysis: {e}")
        return {"score": 0, "suggestions": "An error occurred during AI analysis.", "missing_skills": ["N/A"]}

# --- Text extraction logic ---
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
            resume_data = body.get('resume', None)

            if not job_description_data or not resume_data:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing job description or resume"}).encode('utf-8'))
                return

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

            file_name, content = resume_data.get('fileName'), resume_data.get('content')
            header, base64_content = content.split(',', 1)
            file_bytes = base64.b64decode(base64_content)
            
            resume_text = ""
            if file_name.lower().endswith('.pdf'): resume_text = extract_text_from_pdf(file_bytes)
            elif file_name.lower().endswith('.docx'): resume_text = extract_text_from_docx(file_bytes)

            if not resume_text:
                raise ValueError("Could not read resume text.")

            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', resume_text, re.IGNORECASE)
            candidate_email = email_match.group(0) if email_match else "N/A"

            # --- Single, intelligent AI call ---
            ai_results = get_ai_analysis(resume_text, job_description_text)
            
            total_score = ai_results["score"]
            verdict = "High" if total_score >= 80 else "Medium" if total_score >= 50 else "Low"
            
            result = {
                "resumeName": file_name,
                "candidateEmail": candidate_email,
                "score": total_score,
                "verdict": verdict,
                "missingSkills": ai_results["missing_skills"],
                "suggestions": ai_results["suggestions"]
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"An internal server error occurred: {str(e)}"}).encode('utf-8'))

