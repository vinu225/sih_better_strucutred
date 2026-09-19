# SatQuery AI: Multimodal Earth Observation VLM & AI Agent Framework

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?style=flat&logo=React&logoColor=black)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=Docker&logoColor=white)](https://www.docker.com/)

SatQuery AI is a state-of-the-art Earth Observation platform combining **Multispectral & Multimodal Vision-Language Models (VLM)** with an **Autonomous AI Agent** to analyze Sentinel-2 (Optical) and Sentinel-1 (SAR) satellite imagery.

---

## 📁 Repository Structure

The repository is organized into four core top-level directories:

```text
.
├── backend/                  # FastAPI Python backend, AI Agent, VLM models & pipelines
│   ├── agent/                # Autonomous ReAct agent and satellite tool registry
│   ├── data/                 # Sentinel-2/1 loaders, preprocessing & synthetic generators
│   ├── models/               # Vision encoder, projection layers & Qwen LLM integration
│   ├── services/             # Spectral analysis, SAR processing & classification services
│   ├── tests/                # Comprehensive test suite (80+ unit and integration tests)
│   ├── main.py               # Application entrypoint with OpenAPI configuration
│   ├── routes.py             # REST API endpoints
│   ├── schemas.py            # Pydantic request/response models
│   ├── config.py             # Global constants, band mappings & hyperparams
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # Modern React + Vite interactive web interface
│   ├── src/                  # React UI components, dashboard, analysis modules & styles
│   ├── public/               # Static assets
│   ├── package.json          # Node dependencies & build scripts
│   ├── vite.config.js        # Vite build configuration & API proxy
│   └── streamlit/            # Streamlit prototype archive
│
├── docker/                   # Containerization and deployment configurations
│   ├── Dockerfile.backend    # Python PyTorch & FastAPI production container
│   ├── Dockerfile.frontend   # React / Nginx high-performance SPA container
│   ├── docker-compose.yml    # Multi-service stack orchestration
│   └── .dockerignore         # Docker build exclusions
│
├── readme/                   # Detailed documentation, EDA reports & training resources
│   ├── SatQuery_AI_BigEarthNet_EDA.md
│   ├── sih_prototype_v5.ipynb
│   ├── reben_training_scripts/
│   └── backend_docs.md
│
├── docker-compose.yml        # Root compose configuration
├── .dockerignore
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Run with Docker Compose (Recommended)

To launch the complete frontend and backend stack in production containers:

```bash
# Start all services
docker compose up --build

# Backend API: http://localhost:8000 (Swagger docs at /docs)
# Frontend UI: http://localhost:3000
```

---

### 2. Run Locally (Development Mode)

#### A. Backend Setup
```bash
# Navigate to backend and install requirements
cd backend
pip install -r requirements.txt

# Start backend server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### B. Frontend Setup
```bash
# Navigate to frontend and install npm packages
cd frontend
npm install

# Start Vite development server
npm run dev
# Open http://localhost:5173 in your browser
```

---

## 🛰️ Core Capabilities & Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/analyze` | `POST` | Agentic multispectral and SAR multimodal router |
| `/api/vlm/query` | `POST` | Natural language VLM visual question answering |
| `/api/agent/chat` | `POST` | Multi-turn conversational agent with tool execution |
| `/api/spectral/analyze` | `POST` | Computes NDVI, NDWI, NDBI and vegetation metrics |
| `/api/classify` | `POST` | 19-class BigEarthNet land-cover classification |
| `/api/sar/analyze` | `POST` | Sentinel-1 SAR polarimetric analysis |
| `/api/tiles` | `GET` | List available sample satellite tiles |
| `/api/tiles/{tile_id}/composite` | `GET` | Render true-color / false-color PNG composite |

---

## 🧪 Testing

To execute the backend test suite:

```bash
pytest backend/tests/ -v
```

---

## 📄 License & Citations
Built for Earth Observation research and AI satellite intelligence.
