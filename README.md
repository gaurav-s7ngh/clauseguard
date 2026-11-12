
# ClauseGuard

AI-powered legal document analysis and clause flagging platform.

ClauseGuard uses Natural Language Processing (NLP) and transformer-based models to identify potentially risky clauses in contracts, Terms of Service, and privacy policies.

## Features

- Upload PDF, DOCX and TXT documents.
- Extract and preprocess legal text.
- Detect predefined legal clause categories.
- Highlight potentially risky clauses.
- Assign risk levels: Low, Medium and High.
- Generate simple-English explanations.
- View document analysis results.
- Export analysis reports as PDF.

## Tech Stack

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS
- shadcn/ui

### Backend
- Python
- FastAPI
- SQLAlchemy
- PostgreSQL

### Machine Learning / NLP
- PyTorch
- Hugging Face Transformers
- RoBERTa / InLegalBERT
- Scikit-learn
- Pandas
- NumPy

## Project Structure

```text
clauseguard/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── schemas/
│   │   └── services/
│   └──
│
├── frontend/
│   └── src/
│
├── ml/
│   ├── data/
│   ├── notebooks/
│   ├── preprocessing/
│   ├── training/
│   └── inference/
│
├── evaluation/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

Install the following:

- Python 3.11 or newer
- Node.js and npm
- PostgreSQL
- Git

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd clauseguard
```

### 2. Set up Python environment

```bash
python -m venv backend/venv
```

Activate it on Windows:

```powershell
backend\venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source backend/venv/bin/activate
```

### 3. Install Python dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```bash
cd backend
uvicorn app.main:app --reload
```

Backend URL:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

### 5. Set up the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

## Machine Learning Setup

The backend and ML modules share the same Python virtual environment and root requirements.txt.

ML scripts should be executed from the project root or with the appropriate working directory configured.

## Dataset

Primary dataset:
- CUAD (Contract Understanding Atticus Dataset)

Secondary dataset:
- OPP-115 (Online Privacy Policies Dataset)

Dataset usage must follow the respective licensing and usage conditions.

## Team

Developed as a four-member academic NLP project.

## Disclaimer

ClauseGuard is an educational document-analysis tool. It does not provide legal advice or replace consultation with a qualified legal professional.
