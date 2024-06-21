import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from fastapi.responses import RedirectResponse, Response
from io import BytesIO
from pydantic import BaseModel, Field
import openai
import ast
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the variables from .env
app=FastAPI()


class QuestionRequest(BaseModel):
    role: str
    company: str

@app.post("/generate_questions")
async def generate_questions(request: QuestionRequest):
    role = request.role
    company = request.company

    sys_prompt = """
    You are a helpful assistant that can conduct mock interviews for different job roles at various companies. The user will provide the job role and the company, and you will simulate a mock interview for that role and company using the 'conduct_interview' tool.

    When the user specifies a role and company, use the 'conduct_interview' tool with the 'role' and 'company' arguments set to the provided values. The tool will simulate a mock interview based on the given role and company.

    Respond politely and provide the mock interview simulation output from the tool.
    """

    prompt = f"""Generate a python list of 4 interview questions, enclosed in '[]' with 60 word limit as maximum limit tailored for the role of {role} at {company}. The questions should be relevant to the responsibilities, skills, and qualifications required for the role, as well as the company's culture and values. 
    Present the questions strictly as a Python list, with each question as a separate string element enclosed in double quotes "" within the list."""

    msg = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    openaiclient = openai.Client(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL'),
    )
    resp = openaiclient.chat.completions.create(
        messages=msg,
        temperature=0.6,
        model=os.getenv('OPENAI_MODEL'),
        max_tokens=350,
        stream=False,
    )

    question_list = resp.choices[0].message.content

    start_index = question_list.find('[')
    end_index = question_list.find(']') + 1

    bracket_substring = question_list[start_index:end_index]
    questions = ast.literal_eval(bracket_substring)

    return {"questions": questions}

#For the Interview Tool in AvA
@app.post("/generate_pdf")
async def generate_pdf(
    role: str,
    company: Optional[str] = None,
    questions: List[str] = None,
    answer_list: List[str] = None,
    feedback_list: List[str] = None,
    marks_list: List[int] = None,
):
    if not questions or not answer_list or not feedback_list or not marks_list:
        return {"error": "All input parameters (questions, answer_list, feedback_list, marks_list) are required."}

    max_score = len(questions) * 5
    report_filename = f"{role}_{company or ''}_interview_report.pdf"

    # Create an in-memory buffer for the PDF file
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph(f"Mock Interview Report for {role} at {company or ''}", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    for i, question in enumerate(questions, start=1):
        elements.append(Paragraph(f"Question {i}: {question}", styles["Heading2"]))
        elements.append(Paragraph(f"Your answer: {answer_list[i-1]}", styles["BodyText"]))
        elements.append(Paragraph(f"Evaluation: {feedback_list[i-1]}", styles["BodyText"]))
        elements.append(Paragraph(f"Marks: {marks_list[i-1]}/5", styles["BodyText"]))
        elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Total Score: {sum(marks_list)}/{max_score}", styles["Heading2"]))

    # Build the PDF and write it to the buffer
    doc.build(elements)
    pdf_value = pdf_buffer.getvalue()
    pdf_buffer.close()
    headers = {"Content-Disposition": f"attachment; filename={report_filename}"}
    response = Response(content=pdf_value, media_type="application/pdf", headers=headers)
    return response

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    role: str
    company: str

class EvaluationResponse(BaseModel):
    feedback: str
    marks: int

@app.post("/evaluate_answer", response_model=EvaluationResponse)
async def evaluate_answer_endpoint(request: EvaluationRequest):
    result = evaluate_answer(request.question, request.answer, request.role, request.company)
    return EvaluationResponse(**result)

def evaluate_answer(question: str, answer: str, role: str, company: str) -> dict:
    sys_prompt = """
    You are a helpful assistant that can conduct mock interviews for different job roles at various companies. The user will provide the job role and the company, and you will simulate a mock interview for that role and company using the 'conduct_interview' tool.

    When the user specifies a role and company, use the 'conduct_interview' tool with the 'role' and 'company' arguments set to the provided values. The tool will simulate a mock interview based on the given role and company.

    Respond politely and provide the mock interview simulation output from the tool.
    """
    prompt = f"""Evaluate the following answer for the role of {role} at {company} and for the 

    question:{question}:

    Answer: {answer} 

    Provide feedback on the answer's quality, relevance, and appropriateness, and assign a score to '<score>' from 0 to 5, where 0 is the lowest and 5 is the highest and assign feedback to '<feedback>'.
        provide the response strictly in the following JSON format with the score and feedback assigned:
                    {{
                        "score": "<score>",
                        "feedback": "<feedback>"
                    }}
        No tool calls should be done.
        """

    syss = sys_prompt
    prompt_n = prompt
    msg = [
        {"role": "system", "content": syss},
        {"role": "user", "content": prompt_n},
    ]

    openaiclient = openai.Client(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL'),
    )
    response = openaiclient.chat.completions.create(
        messages=msg,
        temperature=0.6,
        model=os.getenv('OPENAI_MODEL'),
        max_tokens=350,
        stream=False,
    )

    response = response.choices[0].message.content

    try:
        # Extract the JSON object from the response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        json_str = response[json_start:json_end]
        evaluation_json = json.loads(json_str)
        return {
            'feedback': evaluation_json.get('feedback', ''),
            'marks': int(evaluation_json.get('score', 0))
        }
    except (json.JSONDecodeError, ValueError):
        return {'feedback': 'Error: Unable to parse the evaluation response', 'marks': 0}


# Swagger UI is available at http://localhost:8000/docs
@app.get("/", response_class=RedirectResponse)
def redirect_to_docs():
    return "/docs"