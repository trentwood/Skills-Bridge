# Job Architecture System - Quick Start Guide

## 📦 What You Have

A complete job architecture system with **18,328 job titles** from the SOC dataset, organized into:
- **~13,000 unique normalized titles**
- **12+ job families** (Engineering, Data, Product, Sales, etc.)
- **10 organizational levels** (Intern → C-Suite)
- **750+ SOC occupational categories**

## 🚀 Setup (One Time)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Process the SOC Data
```bash
jupyter notebook job_architecture_with_soc.ipynb
```

**Run all cells** in the notebook. This will:
1. Load your SOC_titles.csv file
2. Classify 18K+ titles into levels and families
3. Build the job architecture graph
4. Generate semantic embeddings (takes ~5-10 minutes)
5. Save everything to `job_architecture_data/`

**Expected Output:**
- `job_architecture_data/job_graph.json` (~150MB)
- `job_architecture_data/normalizer_data.pkl` (~500MB with embeddings)
- `job_architecture_data/statistics.json`

### Step 3: Start the Service
```bash
./start_service.sh
# or
python job_architecture_service.py
```

Service runs on: **http://localhost:5001**

### Step 4: Test It
```bash
python test_job_api.py
```

## 🔥 Quick Examples

### Normalize a Job Title
```bash
curl -X POST http://localhost:5001/normalize \
  -H "Content-Type: application/json" \
  -d '{"title": "ML Engineer", "top_k": 3}'
```

### Get Career Path
```bash
curl -X POST http://localhost:5001/career-path \
  -H "Content-Type: application/json" \
  -d '{"title": "Senior Software Engineer", "direction": "up"}'
```

### Search Jobs
```bash
curl -X POST http://localhost:5001/search \
  -H "Content-Type: application/json" \
  -d '{"family": "Engineering", "min_level": 3, "max_level": 4, "limit": 10}'
```

### Get All Families
```bash
curl http://localhost:5001/families
```

## 📊 Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check service status |
| `/stats` | GET | System statistics |
| `/normalize` | POST | Normalize job titles |
| `/career-path` | POST | Get career progression |
| `/search` | POST | Search by criteria |
| `/families` | GET | List job families |
| `/levels` | GET | List org levels |

## 🎯 Common Use Cases

### 1. Resume/Job Posting Normalization
```python
import requests

response = requests.post('http://localhost:5001/normalize', 
    json={"title": "Sr. Software Dev", "top_k": 1})
normalized = response.json()[0]
print(f"Normalized: {normalized['title']}")
print(f"Level: {normalized['level']}, Family: {normalized['family']}")
```

### 2. Career Planning
```python
response = requests.post('http://localhost:5001/career-path',
    json={"title": "Data Scientist", "direction": "up", "limit": 5})
path = response.json()
for job in path['path']:
    print(f"→ {job['title']} (Level {job['level']})")
```

### 3. Organization Design
```python
# Get all engineering leadership roles
response = requests.post('http://localhost:5001/search',
    json={"family": "Engineering", "min_level": 4, "max_level": 6})
results = response.json()
print(f"Found {results['total']} engineering leadership roles")
```

## 📈 Performance Tips

- **First query is slow**: Model loads on first request (~10-30 seconds)
- **Subsequent queries are fast**: ~100-200ms
- **Memory usage**: ~2-3GB (includes embeddings)
- **For production**: Add Redis caching, use FAISS for similarity search

## 🔧 Troubleshooting

### "Data directory not found"
→ Run the Jupyter notebook first to generate data files

### "Port 5001 already in use"
→ Stop existing service or change port in `job_architecture_service.py`

### Out of memory during notebook
→ Process in smaller batches, reduce embedding batch_size to 128

### Slow normalization
→ First query after startup is slower (model loading)
→ Consider caching for production use

## 📁 File Structure

```
job_architecture/
├── job_architecture_with_soc.ipynb   # Data processing notebook
├── job_architecture_service.py       # Flask API service
├── test_job_api.py                   # Comprehensive tests
├── start_service.sh                  # Startup script
├── requirements.txt                  # Python dependencies
├── README.md                         # Full documentation
└── job_architecture_data/            # Generated data (after notebook)
    ├── job_graph.json
    ├── normalizer_data.pkl
    ├── statistics.json
    ├── sample_processed_titles.csv
    └── family_level_summary.csv
```

## 🎨 Customization

### Adjust Level Classification
Edit `JobLevelClassifier` in the notebook to change level patterns

### Add Skills Mapping
Integrate with your skill taxonomy by mapping skills to job IDs

### Industry Filtering
Filter job families and levels by industry and company size

### Custom Families
Modify `JobFamilyClassifier` to add/change job family categories

## 🔗 Integration

### With Skill Extraction Service
Both services can run together:
```bash
# Terminal 1: Skill extraction (port 5000)
cd /path/to/skills
python skill_extraction_service.py

# Terminal 2: Job architecture (port 5001)
cd /path/to/job_architecture
python job_architecture_service.py
```

### In Your Application
```python
import requests

class JobArchitectureClient:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
    
    def normalize(self, title: str, top_k: int = 5):
        response = requests.post(f"{self.base_url}/normalize",
            json={"title": title, "top_k": top_k})
        return response.json()
    
    def career_path(self, title: str, direction: str = "up"):
        response = requests.post(f"{self.base_url}/career-path",
            json={"title": title, "direction": direction})
        return response.json()

# Usage
client = JobArchitectureClient()
matches = client.normalize("Software Developer")
print(matches[0]['title'])
```

## 📚 Next Steps

1. ✅ Run the notebook to process your SOC data
2. ✅ Start the service
3. ✅ Run the test suite
4. 🎯 Integrate with your application
5. 🎯 Add custom skills mapping
6. 🎯 Deploy to production with caching

## 💡 Tips

- **Batch processing**: Process multiple titles in a loop for efficiency
- **Caching**: Cache normalized results for common titles
- **Monitoring**: Add logging and metrics for production use
- **Scaling**: Use FAISS for similarity search at scale (>100K titles)

For detailed documentation, see **README.md**
