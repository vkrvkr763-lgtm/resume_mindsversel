#Resume Relevance Matcher
An AI-powered tool to intelligently score, rank, and provide feedback on resumes against job descriptions, streamlining the recruitment process.

#Problem Statement
Recruiters and hiring managers often face the overwhelming task of sifting through hundreds of resumes for a single job opening. This manual screening process is not only time-consuming and inefficient but also prone to human bias, which can lead to overlooking qualified candidates.

#Our Solution & Approach
This tool automates and enhances the initial screening phase by providing a data-driven analysis of how well a candidate's resume matches a job description.

Our approach is centered around a dual-scoring system to ensure a balanced and comprehensive evaluation:

Keyword-Based Matching (Hard Score): The application first performs a traditional analysis, identifying key skills and qualifications from the job description that are explicitly present in the resume. This provides a foundational score based on direct matches.

AI-Powered Semantic Analysis (Semantic Score): Using Google's Gemini Large Language Model, the tool goes beyond keywords to analyze the contextual relevance of a candidate's experience. It understands industry-specific terminology and can identify relevant skills even if they are not phrased exactly as in the job description.

AI-Generated Feedback: In addition to scoring, the LLM provides actionable suggestions for each candidate, highlighting missing skills and offering advice on how to improve their resume for the specific role.

The final result is a ranked list of candidates, allowing recruiters to focus their time and energy on the most promising applicants.

#Technology Stack
Backend: Python, Flask, LangChain, Google Gemini API

Frontend: HTML, Tailwind CSS, JavaScript

File Processing: PyMuPDF, python-docx

#Getting Started
Follow these steps to set up and run the project locally.

1. Clone the Repository

git clone [https://github.com/your-username/resume-relevance-matcher.git](https://github.com/your-username/resume-relevance-matcher.git)
cd resume-relevance-matcher

2. Create and Activate a Python Virtual Environment

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up Environment Variables

Create a file named .env in the root of the project directory.

Add your Google Gemini API key to this file. You can obtain a key from Google AI Studio.

GOOGLE_API_KEY="your_google_api_key_here"

# How to Run the Application
1. Start the Backend Server

python app.py

The Flask server will start, typically on http://localhost:5000.

2. Open the Frontend
Simply open the index.html file in your web browser.

#Usage Guide
Process Job Description: Upload a JD file or paste the text and click Process.

Upload Resumes: Once the JD is processed, add one or more candidate resumes.

Analyze Resumes: Click Analyze All Resumes to begin the evaluation.

Review Results: The dashboard will populate with a ranked list of candidates, including their scores, verdicts, missing skills, and AI-powered suggestions for improvement.
