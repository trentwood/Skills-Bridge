# 📁 Job Architecture System - File Index

## 🚀 Start Here

1. **PROJECT_SUMMARY.md** - Read this first for complete overview
2. **QUICKSTART.md** - Follow this for step-by-step setup (5 minutes)
3. **README.md** - Full technical documentation and API reference

## 📂 Core Files

### Data Processing
- **job_architecture_with_soc.ipynb** - Jupyter notebook to process your 18K SOC titles
  - Loads SOC_titles.csv
  - Classifies levels and families
  - Builds graph database
  - Generates embeddings
  - Saves to `job_architecture_data/`

### Web Service
- **job_architecture_service.py** - Flask REST API server
  - Runs on port 5001
  - 7 endpoints for normalization, search, career paths
  - Production-ready

### Testing & Validation
- **test_job_api.py** - Comprehensive test suite
  - Tests all endpoints
  - Validates accuracy
  - Example usage patterns

### Data Exploration
- **explore_data.py** - Visualization and data export
  - Creates 5 PNG charts
  - Exports 2 CSV files
  - Generates summary statistics

### Utilities
- **start_service.sh** - One-command service startup
- **requirements.txt** - Python dependencies

## 📋 Step-by-Step Workflow

### First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process the SOC data (10-15 minutes)
jupyter notebook job_architecture_with_soc.ipynb
# → Run all cells

# 3. Start the service
./start_service.sh
# → Service available at http://localhost:5001

# 4. Test it
python test_job_api.py

# 5. Explore the data
python explore_data.py
```

## 📊 What Gets Generated

After running the notebook, you'll have:
```
job_architecture_data/
├── job_graph.json              # ~150MB - Full job graph
├── normalizer_data.pkl         # ~500MB - Embeddings
├── statistics.json             # System stats
├── sample_processed_titles.csv # Sample data
└── family_level_summary.csv    # Distribution matrix
```

After running explore_data.py:
```
├── summary_dashboard.png       # System overview
├── level_distribution.png      # Jobs by level
├── family_distribution.png     # Jobs by family
├── family_level_heatmap.png    # 2D matrix
├── soc_categories.png          # Top SOC categories
├── job_samples_by_level.csv    # Sample export
└── family_summary.csv          # Family stats
```

## 🎯 Quick Reference

### File Sizes
- Total package: ~90KB (source files)
- Generated data: ~650MB (after processing)
- Memory usage: ~2-3GB (when running)

### Processing Time
- Data processing: 10-15 minutes
- Service startup: 10-30 seconds
- Single query: 100-200ms

### Dataset Stats
- Input: 18,328 SOC job titles
- Output: ~13,000 normalized titles
- Searchable: ~70,000 variations
- Families: 12+
- Levels: 10 (0-9)
- SOC Categories: 750+

## 📖 Documentation Guide

### For Quick Setup
1. **QUICKSTART.md** - Step-by-step instructions
2. **start_service.sh** - Run the service

### For Development
1. **README.md** - Full API reference
2. **job_architecture_with_soc.ipynb** - Code walkthrough
3. **test_job_api.py** - Integration examples

### For Understanding
1. **PROJECT_SUMMARY.md** - Complete overview
2. **explore_data.py** - Data analysis

## 🔗 Integration Points

### With Skill Extraction Service
Both services run independently:
- Skill Extraction: Port 5000
- Job Architecture: Port 5001

Call both from your application as needed.

### API Endpoints
- `GET /health` - Service status
- `GET /stats` - Statistics
- `POST /normalize` - Normalize titles
- `POST /career-path` - Career progression
- `POST /search` - Search jobs
- `GET /families` - List families
- `GET /levels` - List levels

See README.md for full API documentation.

## 🎓 Learning Path

### Beginner
1. Read PROJECT_SUMMARY.md
2. Follow QUICKSTART.md
3. Run test_job_api.py
4. Experiment with API

### Intermediate
1. Study job_architecture_with_soc.ipynb
2. Run explore_data.py
3. Modify job_architecture_service.py
4. Add custom features

### Advanced
1. Integrate with your application
2. Customize classification logic
3. Add caching layer
4. Deploy to production

## 🆘 Troubleshooting

### Can't find data files?
→ Run the Jupyter notebook first

### Port already in use?
→ Change port in job_architecture_service.py

### Out of memory?
→ Reduce batch_size in notebook, close other apps

### Slow queries?
→ First query after startup is slower (model loading)

See README.md for more troubleshooting tips.

## 📞 Support

For issues or questions:
1. Check QUICKSTART.md troubleshooting section
2. Review README.md for detailed explanations
3. Examine test_job_api.py for usage examples

## 🎉 You're Ready!

Everything you need is in this package:
- ✅ Data processing pipeline
- ✅ REST API service
- ✅ Comprehensive tests
- ✅ Data exploration tools
- ✅ Complete documentation

**Next step**: Open QUICKSTART.md and follow the setup instructions!
