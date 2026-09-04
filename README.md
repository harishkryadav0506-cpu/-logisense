<p align="center">
  <h1 align="center">🚚 LogiSense</h1>
  <p align="center">
    <strong>Autonomous E-commerce Logistics Copilot</strong>
  </p>
  <p align="center">
    AI-powered complaint resolution system combining RAG, Fine-tuned BERT, and Multi-Agent orchestration
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/LangChain-🦜-green" alt="LangChain">
    <img src="https://img.shields.io/badge/LangGraph-Multi--Agent-blue" alt="LangGraph">
    <img src="https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/BERT-Fine--tuned-orange" alt="BERT">
    <img src="https://img.shields.io/badge/ChromaDB-Vector--Store-purple" alt="ChromaDB">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  </p>
</p>

---

## 📋 Problem Statement

E-commerce companies handle **thousands of customer complaints daily** — delayed deliveries, wrong items, refund requests, and more. Manual resolution is:

- ⏱️ **Slow** — Average 24-48 hours per case
- 🔄 **Inconsistent** — Different agents give different answers
- 💰 **Expensive** — Large support teams needed
- 😤 **Frustrating** — Customers wait in queues

**LogiSense** solves this by deploying an autonomous AI copilot that:

1. **Retrieves** relevant policies using RAG (Retrieval-Augmented Generation)
2. **Classifies** complaint severity using a fine-tuned BERT model
3. **Orchestrates** multiple AI agents to decide and execute the best resolution
4. **Responds** with professional email drafts in seconds, not hours

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                         │
│              (Customer Complaint Input Form)                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │ POST /resolve
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                            │
│              (REST API + Request Validation)                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              LANGGRAPH ORCHESTRATOR                             │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ TRACKER  │──▶│   RAG    │──▶│SENTIMENT │──▶│ RESOLVER │   │
│  │  AGENT   │   │  AGENT   │   │  AGENT   │   │  AGENT   │   │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   │
│       │              │              │              │           │
│       ▼              ▼              ▼              ▼           │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │Order DB │   │ ChromaDB │   │Fine-tuned│   │Refund +  │   │
│  │  Tool   │   │Vector    │   │  BERT    │   │Email Tool│   │
│  │         │   │  Store   │   │  Model   │   │          │   │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                                 │
│  Pipeline: tracker → rag → sentiment → resolver                │
│  Conditional: order_not_found → error_handler → END            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM Orchestration** | LangGraph + LangChain | Multi-agent pipeline with state management |
| **RAG** | ChromaDB + all-MiniLM-L6-v2 | Policy document retrieval |
| **Fine-tuning** | BERT (bert-base-uncased) | Complaint severity classification |
| **Backend** | FastAPI + Uvicorn | REST API with async support |
| **Frontend** | Streamlit | Interactive dashboard UI |
| **Embeddings** | Sentence-Transformers | Document and query embedding |
| **Data** | Pandas + CSV | Order and complaint data management |
| **PDF Processing** | PyPDF | Policy document parsing |
| **LLM Provider** | OpenAI (GPT-3.5/4) | Resolution reasoning (with rule-based fallback) |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip or virtualenv
- (Optional) OpenAI API key for LLM-powered resolution

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/logisense.git
cd logisense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

---

## 🚀 How to Run

### Step 1: Generate Synthetic Data

```bash
python generate_data.py
```

This creates:
- `data/orders.csv` — 1200 orders
- `data/reviews.csv` — 600 complaints
- `data/policies/` — 3 policy PDFs

### Step 2: Initialize RAG Vector Store

```bash
python -m rag.ingest
```

### Step 3: (Optional) Fine-tune BERT Model

```bash
python -m finetuning.train --epochs 3
python -m finetuning.evaluate
```

> **Note:** Training on CPU takes ~15-20 minutes. Use Google Colab with GPU for faster training. The sentiment agent will use keyword-based fallback if the model is not trained.

### Step 4: Start the Backend

```bash
cd logisense
uvicorn backend.main:app --reload --port 8000
```

API available at: http://localhost:8000
API docs at: http://localhost:8000/docs

### Step 5: Start the Frontend

```bash
# In a new terminal
cd logisense
streamlit run frontend/app.py
```

Dashboard available at: http://localhost:8501

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_tools.py -v
pytest tests/test_rag.py -v
pytest tests/test_agents.py -v
```

---

## 📁 Project Structure

```
logisense/
├── data/                          # Synthetic datasets
│   ├── orders.csv                 # 1200 orders with status, carriers, delays
│   ├── reviews.csv                # 600 complaints with severity labels
│   └── policies/                  # Policy PDFs for RAG
│       ├── refund_policy.pdf
│       ├── shipping_sla.pdf
│       └── return_policy.pdf
│
├── rag/                           # RAG Pipeline
│   ├── embeddings.py              # HuggingFace embedding model (MiniLM)
│   ├── ingest.py                  # PDF → chunk → embed → ChromaDB
│   ├── retriever.py               # Semantic search over policies
│   └── vector_store/              # ChromaDB persistent storage
│
├── finetuning/                    # BERT Fine-tuning
│   ├── dataset.py                 # Data preparation + PyTorch Dataset
│   ├── train.py                   # Training with HF Trainer API
│   ├── evaluate.py                # Metrics, confusion matrix, samples
│   └── saved_model/               # Trained model checkpoint
│
├── agents/                        # LangGraph AI Agents
│   ├── orchestrator.py            # StateGraph pipeline (main entry)
│   ├── tracker_agent.py           # Order status lookup
│   ├── rag_agent.py               # Policy document search
│   ├── sentiment_agent.py         # Complaint severity classification
│   └── resolver_agent.py          # Resolution decision + execution
│
├── tools/                         # Function Calling Tools
│   ├── order_db.py                # Order database queries
│   ├── refund_tool.py             # Refund processing + audit log
│   └── email_tool.py              # Email draft generation
│
├── backend/                       # FastAPI Backend
│   └── main.py                    # REST API endpoints
│
├── frontend/                      # Streamlit Frontend
│   └── app.py                     # Interactive dashboard
│
├── tests/                         # Test Suite
│   ├── test_rag.py                # RAG pipeline tests
│   ├── test_tools.py              # Tool function tests
│   └── test_agents.py             # Agent + orchestrator tests
│
├── generate_data.py               # Synthetic data generator
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 📸 Screenshots

> *Screenshots will be added after deployment*

| Dashboard | Resolution Result | Agent Trace |
|-----------|------------------|-------------|
| *Input form with complaint details* | *Resolution with severity badge* | *Step-by-step agent execution* |

---

## 🔮 Future Improvements

- [ ] **Multi-language support** — Handle complaints in multiple languages
- [ ] **Real-time tracking** — WebSocket integration for live order updates
- [ ] **Advanced analytics** — Dashboard with complaint trends and resolution metrics
- [ ] **Voice input** — Speech-to-text for phone-based complaints
- [ ] **Fine-tune on real data** — Replace synthetic data with actual customer complaints
- [ ] **Deployment** — Dockerize and deploy to cloud (AWS/GCP/Azure)
- [ ] **A/B testing** — Compare LLM vs rule-based resolution quality
- [ ] **Feedback loop** — Let customers rate resolutions to improve the model
- [ ] **Multi-turn conversations** — Support follow-up questions from customers
- [ ] **Integration** — Connect with real payment gateways and email services

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 LogiSense

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Built with ❤️ using LangGraph, ChromaDB, BERT & FastAPI
</p>
