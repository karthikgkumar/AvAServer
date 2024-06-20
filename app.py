from typing import List, Optional
from fastapi import FastAPI, File, UploadFile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from fastapi.responses import RedirectResponse, Response
from io import BytesIO

app=FastAPI()


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

# Swagger UI is available at http://localhost:8000/docs
@app.get("/", response_class=RedirectResponse)
def redirect_to_docs():
    return "/docs"