# Skills-Bridge

A comprehensive labor market intelligence platform combining job architecture and skill extraction services.

## Overview

Skills-Bridge provides two complementary microservices:

1. **Job Architecture Service** - Job title normalization, career path analysis, and job search using 18,000+ SOC titles
2. **Skill Extraction Service** - Extract skills from text using semantic similarity with 32,000+ skill taxonomy

Both services use AI-powered semantic matching via sentence transformers for intelligent matching and normalization.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Services

**Job Architecture Service** (Flask, port 5001):
```bash
python src/job_architecture/service.py
```

**Skill Extraction Service** (FastAPI, port 8000):
```bash
uvicorn src.skill_extraction.service:app --host 0.0.0.0 --port 8000
```

Or use Docker Compose to run both:
```bash
docker-compose up
```

### 3. Test the APIs

**Job Architecture:**
```bash
curl -X POST http://localhost:5001/normalize \
  -H "Content-Type: application/json" \
  -d '{"title": "Software Developer", "top_k": 3}'
```

**Skill Extraction:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Experience in Python and machine learning", "max_results": 5}'
```

## Project Structure

```
Skills-Bridge/
├── src/
│   ├── job_architecture/     # Job title normalization service
│   │   └── service.py        # Flask API (port 5001)
│   └── skill_extraction/     # Skill extraction service
│       └── service.py        # FastAPI (port 8000)
├── data/
│   ├── job_architecture/     # Job graph, embeddings, SOC data
│   └── skills/               # Skill taxonomy, embeddings
├── notebooks/                # Jupyter notebooks for data processing
├── tests/                    # API tests
├── scripts/                  # Utility scripts
└── docs/                     # Documentation
```

## Services

### Job Architecture Service

**Endpoints:**
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /normalize` - Normalize job title
- `POST /career-path` - Get career progression
- `POST /search` - Search by criteria
- `GET /families` - List job families
- `GET /levels` - List org levels

**Features:**
- Hybrid matching: exact → fuzzy → semantic
- 18,000+ SOC job titles
- 12 job families, 10 organizational levels
- Career path analysis (up/down/lateral)

### Skill Extraction Service

**Endpoints:**
- `GET /health` - Health check
- `GET /stats` - Service statistics
- `POST /extract` - Extract skills from text

**Features:**
- 32,000+ skills with variations
- N-gram based semantic matching
- Configurable thresholds and deduplication
- Length penalty for partial matches

## Use Cases

### Resume Screening
```python
import requests

# Normalize candidate's job title
job_response = requests.post('http://localhost:5001/normalize', json={
    "title": "ML Eng",
    "top_k": 1
})
normalized = job_response.json()[0]
print(f"Title: {normalized['title']}, Level: {normalized['level']}")

# Extract skills from resume
skill_response = requests.post('http://localhost:8000/extract', json={
    "text": "5 years Python, AWS, machine learning",
    "threshold": 0.6
})
skills = skill_response.json()['skills']
for skill in skills:
    print(f"  - {skill['canonical_name']}")
```

### Career Development
```python
# Get promotion path
response = requests.post('http://localhost:5001/career-path', json={
    "title": "Senior Software Engineer",
    "direction": "up",
    "limit": 5
})
path = response.json()
for job in path['path']:
    print(f"→ {job['title']} (Level {job['level']})")
```

## Data Processing

The raw data is processed using Jupyter notebooks in the `notebooks/` directory:

1. **job_architecture_with_soc.ipynb** - Process SOC titles, build job graph
2. **Skill Taxonomy and Extraction.ipynb** - Build skill taxonomy and embeddings

Run these notebooks to regenerate the data files if needed.

## Performance

- **Job Normalization**: 100-200ms
- **Skill Extraction**: 100-300ms
- **Memory Usage**: ~4-5GB (both services)

## Documentation

- [Job Architecture Details](docs/job_architecture.md)
- [Skill Extraction Details](docs/skill_extraction.md)
- [Quick Start Guide](docs/job_architecture_quickstart.md)

## Technology Stack

- **Frameworks**: Flask, FastAPI
- **ML/NLP**: Sentence Transformers (all-MiniLM-L6-v2)
- **Data**: NetworkX, Pandas, NumPy
- **Matching**: RapidFuzz, Cosine Similarity

## License

MIT License
