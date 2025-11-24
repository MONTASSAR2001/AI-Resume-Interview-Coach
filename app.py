from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from tools import ALL_TOOLS
import PyPDF2

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0.2)

rag_system_prompt = """
You are an expert AI Recruiter using RAG analysis.
You will receive two texts:
1. A Candidate's CV.
2. A Job Description (JD).

Your Task:
1. **Match Score**: Give a percentage score (0-100%) of how well the CV fits the JD.
2. **Gap Analysis**: List skills/keywords present in the JD but MISSING in the CV.
3. **Advice**: Suggest exactly what to add to the CV to increase the match score.
4. **Verdict**: Should they apply? (Yes/No/Maybe).

Be strict but helpful.
"""

agent = create_react_agent(llm, ALL_TOOLS)
app = Flask(__name__)

def extract_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages])
    except: return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    cv_file = request.files['cv']
    job_file = request.files['job']
    
    cv_text = extract_text(cv_file)
    job_text = extract_text(job_file)
    
    if not cv_text or not job_text:
        return jsonify({'response': "❌ Error reading PDF files. Please ensure they are valid text PDFs."})

    rag_prompt = f"""
    Here is the JOB DESCRIPTION:
    {job_text[:10000]}
    
    ----------------
    
    Here is the CANDIDATE CV:
    {cv_text[:10000]}
    
    Please analyze the match based on your instructions.
    """
    
    messages = [
        SystemMessage(content=rag_system_prompt),
        HumanMessage(content=rag_prompt)
    ]
    
    response = agent.invoke({"messages": messages})
    return jsonify({'response': response["messages"][-1].content})

if __name__ == '__main__':
    app.run(debug=True)
