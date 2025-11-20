# Skill Extractor Web Service

A FastAPI-based web service for extracting skills from text using semantic similarity search.

## Features

- **Fast semantic search** using numpy-based similarity computation
- **Smart scoring** with length-based penalties to avoid false positives
- **Deduplication** to remove redundant/similar skills
- **RESTful API** with automatic documentation
- **Configurable parameters** for fine-tuning extraction

## Prerequisites

- Python 3.8+
- Required data files (generated from the Jupyter notebook):
  - `skill_taxonomy.parquet`
  - `skill_variations.parquet`
  - `skill_canonical_embeddings.npy`
  - `skill_variations_embeddings.npy`

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the required data files in the same directory as `skill_extractor_service.py`:
   - `skill_taxonomy.parquet`
   - `skill_variations.parquet`
   - `skill_canonical_embeddings.npy`
   - `skill_variations_embeddings.npy`

## Running the Service

### Basic Usage

```bash
uvicorn skill_extractor_service:app --host 0.0.0.0 --port 8000
```

### With Auto-Reload (Development)

```bash
uvicorn skill_extractor_service:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment

```bash
uvicorn skill_extractor_service:app --host 0.0.0.0 --port 8000 --workers 4
```

The service will be available at: `http://localhost:8000`

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the service is running and initialized.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "skills_loaded": 32468,
  "variations_loaded": 267753
}
```

### 2. Service Statistics

**GET** `/stats`

Get service usage statistics.

**Response:**
```json
{
  "total_requests": 42,
  "total_skills_extracted": 1234,
  "avg_skills_per_request": 29.38,
  "model_name": "all-MiniLM-L6-v2",
  "taxonomy_size": 32468
}
```

### 3. Extract Skills

**POST** `/extract`

Extract skills from text.

**Request Body:**
```json
{
  "text": "I have 5 years of experience in Python programming and machine learning.",
  "threshold": 0.6,
  "top_k": 10,
  "use_variations": true,
  "deduplicate_similar": true,
  "dedup_threshold": 0.85,
  "apply_length_penalty": true,
  "max_results": 10
}
```

**Parameters:**
- `text` (required): Text to extract skills from
- `threshold` (optional, default: 0.6): Minimum similarity threshold (0-1)
- `top_k` (optional, default: 10): Number of top matches to consider per n-gram
- `use_variations` (optional, default: true): Whether to search variations index
- `deduplicate_similar` (optional, default: true): Remove similar/redundant skills
- `dedup_threshold` (optional, default: 0.85): Similarity threshold for deduplication
- `apply_length_penalty` (optional, default: true): Apply penalty for partial matches
- `max_results` (optional): Maximum number of results to return

**Response:**
```json
{
  "skills": [
    {
      "skill_id": "SKILL_12345",
      "canonical_name": "Python Programming",
      "score": 0.892,
      "matched_text": "python programming",
      "matched_variation": "python programming",
      "raw_similarity": 0.945
    },
    {
      "skill_id": "SKILL_23456",
      "canonical_name": "Machine Learning",
      "score": 0.867,
      "matched_text": "machine learning",
      "matched_variation": "ML",
      "raw_similarity": 0.923
    }
  ],
  "total_detected": 2,
  "processing_time_ms": 142.35
}
```

## API Documentation

Once the service is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Using the Test Client

A test client is provided in `test_client.py`:

```bash
python test_client.py
```

Or use it in your own code:

```python
from test_client import SkillExtractorClient

# Create client
client = SkillExtractorClient(base_url="http://localhost:8000")

# Extract skills
result = client.extract_skills(
    text="I have experience in Python and AWS",
    threshold=0.6,
    max_results=10
)

# Print results
for skill in result['skills']:
    print(f"{skill['canonical_name']}: {skill['score']:.3f}")
```

## Using cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Extract Skills
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I have experience in Python programming and machine learning",
    "threshold": 0.6,
    "max_results": 5
  }'
```

## Using Python Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/extract",
    json={
        "text": "I have experience in Python and AWS",
        "threshold": 0.6,
        "use_variations": True,
        "max_results": 10
    }
)

result = response.json()
print(f"Found {result['total_detected']} skills")
for skill in result['skills']:
    print(f"  • {skill['canonical_name']} (score: {skill['score']:.3f})")
```

## Parameter Tuning Guide

### For Higher Precision (fewer false positives)
```json
{
  "threshold": 0.7,
  "dedup_threshold": 0.9,
  "apply_length_penalty": true,
  "use_variations": false
}
```

### For Higher Recall (catch more skills)
```json
{
  "threshold": 0.5,
  "dedup_threshold": 0.8,
  "apply_length_penalty": false,
  "use_variations": true
}
```

### Balanced (recommended starting point)
```json
{
  "threshold": 0.6,
  "dedup_threshold": 0.85,
  "apply_length_penalty": true,
  "use_variations": true
}
```

## Performance

- **Typical response time:** 100-300ms for a resume-length text
- **Throughput:** ~10-20 requests/second on a single worker
- **Memory usage:** ~2-3GB (loaded model + embeddings)

For higher throughput, run with multiple workers:
```bash
uvicorn skill_extractor_service:app --workers 4
```

## Troubleshooting

### Service won't start

**Error:** `FileNotFoundError: Required files not found`

**Solution:** Make sure all required files are in the same directory:
- skill_taxonomy.parquet
- skill_variations.parquet
- skill_canonical_embeddings.npy
- skill_variations_embeddings.npy

### Out of memory

**Error:** Memory error when loading embeddings

**Solution:** 
- Reduce the number of variations per skill in the taxonomy
- Use a machine with more RAM
- Consider using FAISS with quantization (if available)

### Slow response times

**Possible causes:**
- Large input text with many tokens
- Low threshold causing many matches to be evaluated

**Solutions:**
- Increase threshold (e.g., 0.7)
- Reduce `top_k` parameter (e.g., 5)
- Set `max_results` to limit output

## Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY skill_extractor_service.py .
COPY skill_taxonomy.parquet .
COPY skill_variations.parquet .
COPY skill_canonical_embeddings.npy .
COPY skill_variations_embeddings.npy .

EXPOSE 8000

CMD ["uvicorn", "skill_extractor_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t skill-extractor .
docker run -p 8000:8000 skill-extractor
```

## License

[Your license here]

## Support

For issues or questions, please [create an issue](your-repo-url) or contact [your-email].
