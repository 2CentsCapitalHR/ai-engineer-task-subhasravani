# ADGM-Compliant Corporate Agent

An AI-powered legal assistant that reviews, validates, and checks legal documents for compliance with Abu Dhabi Global Market (ADGM) regulations.

This application:
- Accepts `.docx` legal documents.
- Classifies document type (AoA, MoA, Board Resolution, etc.).
- Checks against the ADGM required checklist.
- Detects red flags in document content.
- Inserts inline review notes into the `.docx`.
- Outputs a reviewed document and a structured JSON summary.
- Uses Retrieval-Augmented Generation (RAG) with official ADGM reference documents for accuracy.
- Runs on a Streamlit-based user interface.

---

## Features
- Upload multiple `.docx` legal documents.
- Automatically detect the type of document.
- Compare uploaded documents against ADGM requirements.
- Identify potential compliance issues and red flags.
- Add inline review notes to the `.docx` file.
- Retrieve relevant ADGM rules using embeddings and FAISS.
- Generate outputs in two formats:
  - Reviewed `.docx` with annotations.
  - Structured JSON compliance report.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/2CentsCapitalHR/ai-engineer-task-subhasravani.git
cd ai-engineer-task-subhasravani
