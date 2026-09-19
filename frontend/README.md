# SatQuery AI — Web Frontend (Vite + React + Tailwind CSS)

Modern Earth Observation (EO) conversational interface for SatQuery AI (Smart India Hackathon 2026 Prototype).

---

## 🚀 Quick Start

### 1. Start the FastAPI Backend
In the root `sih_2026_prototype` directory:
```powershell
# Activate Python virtual environment
.\venv\Scripts\activate

# Start backend server
uvicorn satquery.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be active at: `http://localhost:8000/docs`

---

### 2. Install Frontend Dependencies & Start Dev Server
In this directory (`satquery/frontend/web`):
```powershell
npm install
npm run dev
```
Open your browser at: **`http://localhost:5173`**

---

## ⚙️ Environment Variables (Optional)

Create a `.env` file in `satquery/frontend/web/` if needed:
```env
# Custom backend URL if not using default proxy to localhost:8000
VITE_API_URL=http://localhost:8000

# Enable offline mock data demo mode (default: false)
VITE_USE_MOCK=false
```

---

## 📦 Build for Production

```powershell
npm run build
npm run preview
```
