# Hasamex AI Engineer Case Study

## Expert Interview Analysis & Grounded Q&A System

A Streamlit-based AI research application designed to analyze expert interview transcripts across **France, Germany, and the UK**.

The system parses interview transcripts, retrieves relevant evidence using semantic similarity, and uses a grounded Gemini LLM to generate answers based only on the retrieved transcript evidence.

---

## 📌 Project Overview

This project was developed as part of the **Hasamex AI Engineer technical case study**.

The application helps researchers analyze multiple expert interviews and quickly answer questions about topics such as:

* Market adoption
* Adoption barriers
* Economics and ROI
* Training requirements
* Clinical outcomes
* Utilisation
* Growth trends
* Purchasing timelines
* Cross-country differences

The system also preserves the original **expert name, timestamp, country, and transcript statement** so that generated answers can be traced back to the source evidence.

---

## 🎯 Problem Statement

Analyzing multiple expert interviews manually can be time-consuming.

Researchers need to:

1. Read multiple transcripts.
2. Identify relevant expert statements.
3. Compare findings across countries.
4. Answer interview-guide questions.
5. Locate supporting quotes and timestamps.
6. Avoid introducing information that was not present in the interviews.

This project provides a lightweight AI-assisted workflow for performing these tasks more efficiently.

---

## ✨ Key Features

### 1. Interview Guide

Displays the questions extracted from the provided interview guide.

The application currently processes **6 interview-guide questions**.

---

### 2. Evidence Retrieval

Relevant statements can be identified from the expert transcripts using:

* Keyword-based evidence retrieval
* Semantic similarity retrieval
* Expert-only filtering

Interviewer statements are excluded from the semantic evidence retrieval pipeline.

---

### 3. Semantic Search

The application uses **Sentence Transformers** to convert transcript statements and user questions into embeddings.

The current embedding model is:

```text
all-MiniLM-L6-v2
```

Cosine similarity is used to identify relevant transcript statements.

---

### 4. Grounded AI Q&A

Users can ask natural-language questions across the three transcripts.

Example:

```text
What are the main barriers to adoption?
```

The system:

```text
User Question
      ↓
Semantic Retrieval
      ↓
Relevant Transcript Evidence
      ↓
Grounded Prompt
      ↓
Gemini
      ↓
Evidence-based Answer
```

The Gemini prompt explicitly instructs the model to:

* Use only retrieved transcript evidence
* Avoid outside knowledge
* Avoid inventing facts
* Preserve numbers from the transcript
* Keep countries separate
* Identify insufficient evidence
* Provide evidence references

---

### 5. Evidence References

Retrieved evidence includes:

```text
Country
Expert
Timestamp
Transcript statement
Semantic relevance score
```

This allows the user to inspect the source statements behind the answer.

---

### 6. Cross-Country Analysis

The application provides analysis across:

* 🇫🇷 France
* 🇩🇪 Germany
* 🇬🇧 UK

It can organize evidence by country and identify common themes and areas where evidence differs.

---

### 7. Theme Extraction

The application identifies transcript evidence related to predefined research themes.

Current themes include:

* Economics & Cost
* Training & Skills
* Clinical Outcomes
* Utilisation
* Adoption & Growth

---

### 8. Hallucination Control

The system follows a **retrieval-grounded generation** approach.

The LLM does not receive the complete unrestricted transcript as its knowledge source for each question.

Instead:

```text
Question
   ↓
Relevant Evidence
   ↓
Grounded Prompt
   ↓
LLM
```

The prompt instructs Gemini to use only the retrieved evidence.

If relevant evidence is unavailable for a country, the system instructs the model to explicitly state that sufficient evidence was not retrieved instead of creating an answer.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │  Expert Transcripts  │
                    │ France / Germany / UK│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Document Loader     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Transcript Parser    │
                    │                      │
                    │ Timestamp            │
                    │ Speaker              │
                    │ Transcript Text      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Semantic Retrieval   │
                    │                      │
                    │ Sentence Transformers│
                    │ all-MiniLM-L6-v2     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Relevant Evidence    │
                    │                      │
                    │ Country              │
                    │ Expert               │
                    │ Timestamp            │
                    │ Text                 │
                    │ Score                │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Grounded Prompt      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Gemini LLM           │
                    │ Grounded Synthesis   │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Answer + Evidence References   │
              └────────────────────────────────┘
```

---

## 🧠 AI / NLP Approach

### Transcript Processing

The transcripts are parsed into structured records:

```python
{
    "timestamp": "...",
    "speaker": "...",
    "text": "..."
}
```

This allows the application to preserve source-level metadata throughout the pipeline.

---

### Semantic Retrieval

The system uses:

```text
SentenceTransformer
        ↓
all-MiniLM-L6-v2
        ↓
Text Embeddings
        ↓
Cosine Similarity
```

The user's question is embedded and compared against expert statements.

The highest-scoring relevant statements are selected as evidence.

---

### Grounded Generation

The retrieved evidence is passed into a structured prompt.

The prompt establishes the following principle:

> The transcript evidence is the source of truth.

Gemini is therefore used primarily for:

* Summarization
* Organization
* Comparison
* Natural-language answer generation

The underlying transcript evidence remains the source of the factual content.

---

## 📂 Project Structure

```text
hasamex-ai-case-study/
│
├── app.py
├── test_semantic.py
├── test_llm.py
│
├── data/
│   ├── Transcript_1_France.txt
│   ├── Transcript_2_Germany.txt
│   ├── Transcript_3_UK.txt
│   └── Interview_Guide.txt
│
└── src/
    ├── __init__.py
    ├── document_loader.py
    ├── transcript_parser.py
    ├── guide_parser.py
    ├── evidence_retriever.py
    ├── semantic_retriever.py
    ├── comparison.py
    ├── theme_extractor.py
    ├── qa_engine.py
    ├── llm_prompt.py
    └── llm_engine.py
```

---

## 🛠️ Technology Stack

### Programming

* Python

### Application

* Streamlit

### NLP / Machine Learning

* Sentence Transformers
* Semantic Embeddings
* Cosine Similarity

### Generative AI

* Google Gemini API

### Data Processing

* Python
* Regular Expressions
* Structured dictionaries / lists

### UI

* Streamlit
* Custom CSS styling

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd hasamex-ai-case-study
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit sentence-transformers google-genai
```

If a `requirements.txt` file is included in the repository:

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API

Set the Gemini API key as an environment variable.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Do **not** commit the API key to GitHub.

---

## ▶️ Run the Application

From the project directory:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 🔍 Example Questions

The application can be used to ask questions such as:

```text
What are the main barriers to adoption?
```

```text
How does training affect adoption?
```

```text
What factors influence ROI and purchasing decisions?
```

```text
What are the expected adoption trends?
```

```text
How long does the purchasing process take?
```

```text
What is the market share of robotic surgery in France, Germany, and the UK?
```

For questions where the provided transcripts do not contain sufficient information, the grounded pipeline is designed to indicate that the evidence is insufficient rather than inventing information.

---

## 🛡️ Handling Unsupported Questions

A key design consideration is avoiding unsupported AI-generated claims.

For example, if the transcripts discuss:

```text
Adoption
Procedure volume
Costs
Training
Purchasing timelines
```

but do not provide:

```text
Market share percentages
```

the system should not manufacture market-share values.

The application instead relies on the retrieved evidence and the grounding instructions given to the LLM.

---

## 📊 Current Dataset

The current case study contains:

| Component                 |               Count |
| ------------------------- | ------------------: |
| Expert transcripts        |                   3 |
| Countries                 |                   3 |
| Interview guide questions |                   6 |
| Countries covered         | France, Germany, UK |

---

## 🌍 Country Coverage

### France

Expert interview evidence related to adoption, capital budgets, economics, utilisation, training, growth and purchasing timelines.

### Germany

Expert interview evidence related to adoption, cost barriers, hospital finances, utilisation, training, procedure volumes and purchasing decisions.

### UK

Expert interview evidence related to adoption, funding, training, clinical outcomes, growth and purchasing timelines.

---

## 🚦 Graceful API Failure Handling

Gemini is an external API and may become temporarily unavailable because of quota or rate limits.

The application handles Gemini failures without treating the entire retrieval pipeline as failed.

For example:

```text
Semantic Retrieval       ✅
Retrieved Evidence       ✅
Grounded Prompt          ✅
Gemini Generation        ⚠️ Temporarily unavailable
```

When Gemini is unavailable, the application can still display the retrieved transcript evidence.

This separates the **retrieval layer** from the **LLM generation layer**.

---

## 🧪 Testing

The project includes basic test scripts for the main AI components.

### Semantic Retrieval Test

```bash
python test_semantic.py
```

### LLM Test

```bash
python test_llm.py
```

The LLM test requires a valid Gemini API key and available API quota.

---

## ⚠️ Limitations

This is a technical case-study implementation rather than a production research platform.

Current limitations include:

* The dataset contains only three transcripts.
* Semantic retrieval depends on the quality of the embedding model.
* The current retrieval pipeline uses a fixed relevance threshold.
* Gemini API availability depends on external API quota.
* Theme extraction currently uses predefined themes and keywords.
* Cross-country contrast detection is evidence-based but does not perform sophisticated semantic disagreement classification.
* No model fine-tuning is performed.
* Results should be reviewed by a human before being used for formal research decisions.

---

## 🚀 Future Improvements

If this system were expanded beyond the case study, possible improvements would include:

### Retrieval

* Hybrid semantic + keyword retrieval
* Reranking models
* Chunk-level retrieval
* Vector database integration
* Metadata filtering

### Evidence Quality

* Automatic citation validation
* Evidence confidence scoring
* Better contradiction detection
* More advanced cross-country comparison

### Scale

The current architecture can be extended from:

```text
3 transcripts
```

to:

```text
30+
transcripts
```

by introducing persistent vector storage and metadata-based retrieval.

### Production AI

Future versions could include:

* Retrieval-Augmented Generation (RAG)
* Vector databases
* Background document processing
* Batch transcript ingestion
* Evaluation datasets
* Automated hallucina
