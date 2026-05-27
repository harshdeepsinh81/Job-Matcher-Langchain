import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "Job-Matcher"

st.set_page_config(page_title="AI Resume Matcher (LangChain)", layout="wide")
st.title("Job Matcher (LangChain + Ollama)")

llm = ChatOpenAI(
    model="gpt-oss:120b",
    base_url="https://ollama.com/v1",   
    api_key=os.getenv("OLLAMA_API_KEY")
)

prompt = ChatPromptTemplate.from_template(""" 
You are a senior recruiter and ATS expert.

Compare the resume and job description and provide:

1. Job Summary
2. Key Skills Required
3. Matching Skills
4. Missing Skills
5. Strengths
6. Weaknesses
7. ATS Score (0-100)
8. Interview Questions
9. What to Focus On
10. 7-Day Improvement Plan

---

JOB DESCRIPTION:
{job}

---

RESUME:
{resume}
""")

# LANGCHAIN CHAIN
chain = prompt | llm | StrOutputParser()

def scrape_job(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator=" ")[:12000]

def read_resume(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text[:12000]

def analyze(job_text, resume_text):
    return chain.invoke({
        "job": job_text,
        "resume": resume_text
    })

job_url = st.text_input("Paste Job URL")
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

run = st.button("Analyze")

if run:

    if not job_url or not resume_file:
        st.error("Please provide both Job URL and Resume PDF")
        st.stop()

    with st.spinner("Scraping job description..."):
        job_text = scrape_job(job_url)

    with st.spinner("Reading resume..."):
        resume_text = read_resume(resume_file)

    with st.spinner("AI analyzing with LangChain..."):
        result = analyze(job_text, resume_text)

    st.success("Analysis Complete!")

    st.markdown("## 📊 ATS Analysis Report")
    st.write(result)