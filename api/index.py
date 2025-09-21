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

# --- Utility Functions ---
def parse_llm_json_response(response_text: str, default_value=None):
    """Safely extracts and parses a JSON object from a string."""
    json_match = re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL)
    if not json_match:
        print(f"[api] AI response did not contain JSON: {response_text}")
        return default_value or {}
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        print(f"[api] Failed to decode JSON from AI response: {response_text}")
        return default_value or {}

# --- Core AI Functions ---
def extract_jd_skills(jd_text: str) -> list:
    """Uses LLM to extract key skills from a job description."""
    if not llm:
        return []
    try:
        prompt = PromptTemplate.from_template(
            """Analyze the following job description. Extract the most critical skills, technologies, and qualifications.
Respond ONLY with a single, valid JSON array of strings.
Example: ["Python", "Data Analysis", "Machine Learning", "Communication"]

Job Description: {jd}
JSON Array Response:
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response_text = chain.invoke({"jd": jd_text}).get("text", "[]")
        return parse_llm_json_response(response_text, default_value=[])
    except Exception as e:
        print(f"[api] Error in extract_jd_skills: {e}")
        return []

def get_llm_analysis(resume_text: str, jd_text: str) -> dict:
    """Uses LLM to get a contextual score and a 3-line feedback summary."""
    if not llm:
        return {"score": 0, "suggestions": "LLM not available."}
    try:
        prompt = PromptTemplate.from_template(
            """Analyze the resume against the job description. Respond ONLY with a single, valid JSON object with two keys: "score" and "suggestions".
- "score": An integer (0-100) for contextual match quality.
- "suggestions": A concise, constructive 3-line feedback summary for the candidate.

Example of a valid response:
{{"score": 75, "suggestions": "Your experience in data analysis is a good starting point. To better align with this role, consider highlighting projects where you used Python for data manipulation. Emphasizing collaboration with product teams would also strengthen your profile."}}

JD: {jd}
Resume: {resume}
JSON Response:
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response_text = chain.invoke({"resume": resume_text, "jd": jd_text}).get("text", "{}")
        parsed_json = parse_llm_json_response(response_text, default_value={"score": 0, "suggestions": "Error parsing AI feedback."})
        
        # Normalize score to be out of 50
        normalized_score = (int(parsed_json.get("score", 0)) / 100) * 50
        
        return {
            "score": max(0, min(normalized_score, 50)),
            "suggestions": parsed_json.get("suggestions", "No suggestions generated.")
        }
    except Exception as e:
        print(f"[api] Critical error in get_llm_analysis: {e}")
        return {"score": 0, "suggestions": "Error during AI analysis."}

def generate_overall_feedback(results_summary: str) -> str:
    """Uses LLM to generate a 3-line summary of the entire candidate pool."""
    if not llm:
        return "LLM not available for summary generation."
    try:
        prompt = PromptTemplate.from_template(
            """Based on the following summary of candidate analysis results, provide a concise, 3-line overall feedback on the candidate pool.
Mention common strengths and weaknesses.

Analysis Summary: {summary}
Overall Feedback:
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.invoke({"summary": results_summary}).get("text", "Could not generate overall feedback.")
    except Exception as e:
        print(f"[api] Error in generate_overall_feedback: {e}")
        return "Error generating summary."

# --- Text extraction and keyword matching logic ---
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

def get_hard_match_score(resume_text: str, required_skills: list):
    """Calculates a score based on the presence of required skills in the resume."""
    if not required_skills:
        return 0.0, [], []

    resume_text_lower = resume_text.lower()
    matched = [skill for skill in required_skills if re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_text_lower)]
    missing = [skill for skill in required_skills if skill not in matched]
    
    score = (len(matched) / len(required_skills)) * 50 if required_skills else 0
    return score, matched, missing

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
            
            if not job_description_text:
                 raise ValueError("Job description text is empty.")

            # 1. Extract skills from JD once
            jd_skills = extract_jd_skills(job_description_text)

            analysis_results = []
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

                hard_score, matched_skills, missing_skills = get_hard_match_score(resume_text, jd_skills)
                llm_results = get_llm_analysis(resume_text, job_description_text)
                
                total_score = round(hard_score + llm_results["score"])
                verdict = "High" if total_score >= 80 else "Medium" if total_score >= 50 else "Low"
                
                analysis_results.append({
                    "candidateName": file_name,
                    "candidateEmail": candidate_email,
                    "score": total_score,
                    "verdict": verdict,
                    "missingSkills": missing_skills,
                    "suggestions": llm_results["suggestions"]
                })
            
            # 2. Generate overall feedback
            results_summary_for_llm = json.dumps([{"score": r["score"], "missing": r["missingSkills"]} for r in analysis_results])
            overall_feedback = generate_overall_feedback(results_summary_for_llm)

            # 3. Final response structure
            final_response = {
                "results": analysis_results,
                "jobDescriptionSkills": jd_skills,
                "overallFeedback": overall_feedback,
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(final_response).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"An internal server error occurred: {str(e)}"}).encode('utf-8'))

