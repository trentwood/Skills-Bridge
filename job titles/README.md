# Job Architecture System with SOC Titles

A comprehensive job architecture system built on 18,000+ job titles from the Standard Occupational Classification (SOC) system.

## Overview

This system provides:
- **18,000+ normalized job titles** from the US labor market
- **Intelligent job title normalization** using hybrid matching (exact, fuzzy, semantic)
- **Career path analysis** (promotions, lateral moves, direct reports)
- **Job search** by family, level, and other criteria
- **RESTful API** for easy integration
- **Graph-based architecture** for relationship modeling

## Dataset Statistics

- **Total Jobs**: ~13,000 unique normalized titles
- **Searchable Titles**: ~70,000+ (including variations)
- **Job Families**: 12+ (Engineering, Data, Product, Sales, Marketing, etc.)
- **Organizational Levels**: 10 (0=Intern → 9=C-Suite)
- **SOC Categories**: 750+ occupational groups

## Quick Start

### 1. Process the Data

Run the Jupyter notebook to process the SOC titles and build the system:

```bash
jupyter notebook job_architecture_with_soc.ipynb
```

Run all cells to:
1. Load and classify 18K+ job titles
2. Assign organizational levels (0-9)
3. Classify into job families
4. Build the job architecture graph
5. Generate semantic embeddings
6. Save all data to `job_architecture_data/`

### 2. Start the API Service

```bash
python job_architecture_service.py
```

The service will start on `http://localhost:5001`

### 3. Test the API

```bash
python test_job_api.py
```

## API Endpoints

### Health Check
```bash
GET /health
```

Returns service status and basic statistics.

**Response:**
```json
{
  "status": "healthy",
  "service": "job-architecture",
  "stats": {
    "total_jobs": 13252,
    "searchable_titles": 71523,
    "families": 12,
    "levels": 10
  }
}
```

### Normalize Job Title
```bash
POST /normalize
Content-Type: application/json

{
  "title": "Software Developer",
  "top_k": 5,
  "fuzzy_threshold": 80
}
```

**Response:**
```json
[
  {
    "title": "Software Engineer",
    "job_id": "job_00123",
    "level": 2,
    "family": "Engineering",
    "soc_category": "Software Developers",
    "similarity_score": 0.95,
    "match_type": "fuzzy"
  }
]
```

**Match Types:**
- `exact`: Perfect match found
- `fuzzy`: Similar spelling/wording (handles typos, abbreviations)
- `semantic`: Conceptually similar (uses AI embeddings)

### Get Career Path
```bash
POST /career-path
Content-Type: application/json

{
  "title": "Senior Software Engineer",
  "direction": "up",
  "limit": 20
}
```

**Directions:**
- `up`: Promotion path (higher levels in same family)
- `down`: Direct reports (lower levels in same family)
- `lateral`: Peer roles (same level, different families)

**Response:**
```json
{
  "current_job": {
    "title": "Senior Software Engineer",
    "level": 3,
    "family": "Engineering"
  },
  "direction": "up",
  "path": [
    {
      "title": "Staff Software Engineer",
      "level": 4,
      "family": "Engineering"
    },
    {
      "title": "Principal Software Engineer",
      "level": 4,
      "family": "Engineering"
    }
  ],
  "total_available": 15
}
```

### Search Jobs
```bash
POST /search
Content-Type: application/json

{
  "family": "Engineering",
  "min_level": 3,
  "max_level": 4,
  "limit": 50
}
```

**Filters:**
- `family`: Job family (optional)
- `level`: Exact level (optional)
- `min_level`: Minimum level (optional)
- `max_level`: Maximum level (optional)
- `limit`: Max results (default: 50)

### Get Families
```bash
GET /families
```

Returns all job families and their counts.

### Get Levels
```bash
GET /levels
```

Returns organizational level descriptions and counts.

### Get Statistics
```bash
GET /stats
```

Returns detailed system statistics.

## Organizational Levels

| Level | Description | Examples |
|-------|-------------|----------|
| 9 | C-Suite/Executive | CEO, CTO, CFO, President |
| 8 | Senior VP | SVP Engineering, EVP Sales |
| 7 | VP | VP Product, VP Marketing |
| 6 | Director | Director of Engineering, Director of Data Science |
| 5 | Senior Manager | Senior Engineering Manager, Group Manager |
| 4 | Manager/Principal IC | Engineering Manager, Staff Engineer, Principal Scientist |
| 3 | Senior IC | Senior Software Engineer, Senior Data Scientist |
| 2 | Mid-Level IC | Software Engineer, Data Analyst, Product Designer |
| 1 | Junior IC | Junior Engineer, Associate Analyst |
| 0 | Intern/Entry | Software Engineer Intern, Data Science Intern |

## Job Families

- **Engineering**: Software, DevOps, SRE, Infrastructure, Security
- **Data**: Data Science, Analytics, ML Engineering, BI
- **Product**: Product Management, Program Management
- **Design**: UX, UI, Visual Design, Product Design
- **Sales**: Account Executives, Sales Engineers, Business Development
- **Marketing**: Digital Marketing, Brand, Content, Growth
- **HR**: Recruiting, People Operations, Compensation, Learning
- **Finance**: FP&A, Accounting, Treasury, Financial Analysis
- **Operations**: Supply Chain, Logistics, Project Management, Facilities
- **Customer Success**: Support, Account Management, Client Services
- **Legal**: Counsel, Compliance, Regulatory
- **Executive**: C-Suite leadership across all functions

## Example Use Cases

### 1. Resume Screening
```python
import requests

# Normalize a candidate's job title
response = requests.post('http://localhost:5001/normalize', json={
    "title": "ML Eng",
    "top_k": 3
})

matches = response.json()
best_match = matches[0]
print(f"Normalized: {best_match['title']}")
print(f"Level: {best_match['level']}, Family: {best_match['family']}")
```

### 2. Career Development Planning
```python
# Find promotion opportunities
response = requests.post('http://localhost:5001/career-path', json={
    "title": "Product Manager",
    "direction": "up",
    "limit": 10
})

career_path = response.json()
print(f"Next roles for {career_path['current_job']['title']}:")
for job in career_path['path']:
    print(f"  - {job['title']} (Level {job['level']})")
```

### 3. Organization Design
```python
# Find all engineering managers
response = requests.post('http://localhost:5001/search', json={
    "family": "Engineering",
    "min_level": 4,
    "max_level": 6
})

results = response.json()
print(f"Engineering leadership roles: {results['total']}")
```

### 4. Job Board Integration
```python
# Normalize job posting titles for better matching
job_postings = [
    "Sr. Software Developer",
    "Lead ML Engineer", 
    "VP Product",
]

for posting in job_postings:
    response = requests.post('http://localhost:5001/normalize', json={
        "title": posting,
        "top_k": 1
    })
    normalized = response.json()[0]
    print(f"{posting} → {normalized['title']}")
```

## Data Files

After running the notebook, the following files are created in `job_architecture_data/`:

- **job_graph.json**: Complete job architecture with all relationships
- **normalizer_data.pkl**: Pre-computed embeddings and lookup tables
- **statistics.json**: System-wide statistics
- **sample_processed_titles.csv**: Sample of processed data
- **family_level_summary.csv**: Distribution matrix

## Architecture

The system uses:
- **NetworkX**: Graph database for job relationships
- **Sentence Transformers**: Semantic similarity via embeddings
- **RapidFuzz**: Fast fuzzy string matching
- **Flask**: RESTful API server
- **Pandas/NumPy**: Data processing

## Performance

- **Normalization**: ~100-200ms per query (with caching)
- **Career Path**: ~50-100ms per query
- **Search**: ~10-50ms per query (depends on filters)
- **Memory Usage**: ~2-3GB (includes embeddings)

## Customization

### Add Custom Skills
Edit the notebook to map skills to job titles based on your taxonomy.

### Adjust Level Classification
Modify `JobLevelClassifier` patterns to match your organization's structure.

### Industry-Specific Architectures
Filter job families and levels based on industry and company size.

## Integration with Skill Extraction

To integrate with the skill extraction service from the previous project:

```bash
# In one terminal
cd /path/to/skills
python skill_extraction_service.py

# In another terminal
cd /path/to/job-architecture
python job_architecture_service.py
```

Both services can run simultaneously on ports 5000 and 5001.

## Troubleshooting

### Service won't start
- Check that port 5001 is available
- Ensure all data files exist in `job_architecture_data/`
- Verify Python packages are installed: `pip install -r requirements.txt`

### Slow queries
- First query after startup is slower (model loading)
- Consider adding Redis caching for production
- Reduce `top_k` parameter for faster results

### Out of memory
- Reduce batch size in notebook when computing embeddings
- Use a smaller sentence transformer model
- Consider using FAISS for similarity search at scale

## Future Enhancements

- [ ] Add salary data by job level and location
- [ ] Skills taxonomy integration
- [ ] Industry-specific filtering
- [ ] Company size customization
- [ ] Export to Neo4j graph database
- [ ] Real-time job market data updates
- [ ] Multi-language support
- [ ] Authentication and rate limiting
- [ ] Caching layer (Redis)
- [ ] Admin UI for taxonomy management

## Credits

Built on the Standard Occupational Classification (SOC) system from the U.S. Bureau of Labor Statistics.

## License

MIT License - See LICENSE file for details
