from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import sys

app = Flask(__name__)
CORS(app)

# Load data - use path relative to project root
data_dir = Path(__file__).parent.parent.parent / "data" / "job_architecture"

print("="*60)
print("Loading Job Architecture Data...")
print("="*60)

print("\n1. Loading job graph...")
with open(data_dir / "job_graph.json", 'r') as f:
    graph_data = json.load(f)

print("2. Loading normalizer data...")
with open(data_dir / "normalizer_data.pkl", 'rb') as f:
    normalizer_data = pickle.load(f)

print("3. Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("4. Loading statistics...")
with open(data_dir / "statistics.json", 'r') as f:
    stats = json.load(f)

# Extract data
job_lookup = graph_data['job_lookup']
title_to_id = graph_data['title_to_id']
all_titles = normalizer_data['all_titles']
title_to_job = normalizer_data['title_to_job']
title_embeddings = normalizer_data['title_embeddings']

print("="*60)
print("Data Loaded Successfully!")
print("="*60)
print(f"Jobs: {len(job_lookup):,}")
print(f"Searchable Titles: {len(all_titles):,}")
print(f"Families: {len(stats['families'])}")
print(f"Levels: {len(stats['levels'])}")
print("="*60)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "job-architecture",
        "stats": {
            "total_jobs": len(job_lookup),
            "searchable_titles": len(all_titles),
            "families": len(stats['families']),
            "levels": len(stats['levels'])
        }
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify(stats)

@app.route('/normalize', methods=['POST'])
def normalize_title():
    """
    Normalize a job title
    
    Request: {"title": "Software Developer", "top_k": 5}
    Response: [{"title": "...", "job_id": "...", "similarity_score": 0.95, ...}]
    """
    data = request.get_json()
    input_title = data.get('title', '')
    top_k = data.get('top_k', 5)
    fuzzy_threshold = data.get('fuzzy_threshold', 80)
    
    if not input_title:
        return jsonify({"error": "Title is required"}), 400
    
    # Check exact match
    if input_title.lower() in title_to_id:
        job_id = title_to_id[input_title.lower()]
        job = job_lookup[job_id]
        return jsonify([{
            "title": job['title'],
            "job_id": job_id,
            "level": job['level'],
            "family": job['family'],
            "soc_category": job['soc_category'],
            "similarity_score": 1.0,
            "match_type": "exact"
        }])
    
    # Fuzzy matching
    fuzzy_matches = process.extract(
        input_title, 
        all_titles, 
        scorer=fuzz.token_sort_ratio,
        limit=top_k * 2
    )
    
    fuzzy_results = []
    for match_title, score, _ in fuzzy_matches:
        if score >= fuzzy_threshold:
            job = title_to_job[match_title]
            fuzzy_results.append({
                "title": job['title'],
                "job_id": job['id'],
                "level": job['level'],
                "family": job['family'],
                "soc_category": job['soc_category'],
                "similarity_score": score / 100.0,
                "match_type": "fuzzy"
            })
    
    # Semantic similarity
    input_embedding = model.encode([input_title])
    similarities = cosine_similarity(input_embedding, title_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k * 2:][::-1]
    
    semantic_results = []
    for idx in top_indices:
        match_title = all_titles[idx]
        job = title_to_job[match_title]
        semantic_results.append({
            "title": job['title'],
            "job_id": job['id'],
            "level": job['level'],
            "family": job['family'],
            "soc_category": job['soc_category'],
            "similarity_score": float(similarities[idx]),
            "match_type": "semantic"
        })
    
    # Combine and deduplicate
    seen_ids = set()
    combined_results = []
    
    for result_list in [fuzzy_results, semantic_results]:
        for result in result_list:
            if result["job_id"] not in seen_ids:
                seen_ids.add(result["job_id"])
                combined_results.append(result)
    
    combined_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return jsonify(combined_results[:top_k])

@app.route('/career-path', methods=['POST'])
def get_career_path():
    """
    Get career path for a job
    
    Request: {"title": "Software Engineer", "direction": "up", "limit": 20}
    Direction: up (promotions), down (reports), lateral (same level)
    """
    data = request.get_json()
    title = data.get('title', '')
    direction = data.get('direction', 'up')
    limit = data.get('limit', 20)
    
    if not title:
        return jsonify({"error": "Title is required"}), 400
    
    # Normalize title first
    if title.lower() in title_to_id:
        job_id = title_to_id[title.lower()]
    else:
        # Try to find best match
        matches = process.extract(title, all_titles, scorer=fuzz.token_sort_ratio, limit=1)
        if matches and matches[0][1] >= 80:
            match_title = matches[0][0]
            job_id = title_to_job[match_title]['id']
        else:
            return jsonify({"error": "Job title not found"}), 404
    
    current_job = job_lookup[job_id]
    current_level = current_job['level']
    current_family = current_job['family']
    
    # Get career path based on direction
    path = []
    
    if direction == 'up':
        # Higher levels in same family
        for jid, job in job_lookup.items():
            if job['family'] == current_family and job['level'] > current_level:
                path.append({
                    "title": job['title'],
                    "job_id": jid,
                    "level": job['level'],
                    "family": job['family'],
                    "soc_category": job['soc_category']
                })
    elif direction == 'down':
        # Lower levels in same family
        for jid, job in job_lookup.items():
            if job['family'] == current_family and job['level'] < current_level:
                path.append({
                    "title": job['title'],
                    "job_id": jid,
                    "level": job['level'],
                    "family": job['family'],
                    "soc_category": job['soc_category']
                })
    elif direction == 'lateral':
        # Same level, different families
        for jid, job in job_lookup.items():
            if job['level'] == current_level and job['family'] != current_family:
                path.append({
                    "title": job['title'],
                    "job_id": jid,
                    "level": job['level'],
                    "family": job['family'],
                    "soc_category": job['soc_category']
                })
    
    # Sort by level
    path.sort(key=lambda x: x['level'], reverse=(direction == 'up'))
    
    return jsonify({
        "current_job": {
            "title": current_job['title'],
            "job_id": job_id,
            "level": current_level,
            "family": current_family,
            "soc_category": current_job['soc_category']
        },
        "direction": direction,
        "path": path[:limit],
        "total_available": len(path)
    })

@app.route('/search', methods=['POST'])
def search_jobs():
    """
    Search jobs by various criteria
    
    Request: {
        "family": "Engineering",
        "level": 3,
        "min_level": 2,
        "max_level": 4,
        "limit": 50
    }
    """
    data = request.get_json()
    family = data.get('family')
    level = data.get('level')
    min_level = data.get('min_level')
    max_level = data.get('max_level')
    limit = data.get('limit', 50)
    
    results = []
    
    for job_id, job in job_lookup.items():
        # Apply filters
        if family and job['family'] != family:
            continue
        if level is not None and job['level'] != level:
            continue
        if min_level is not None and job['level'] < min_level:
            continue
        if max_level is not None and job['level'] > max_level:
            continue
        
        results.append({
            "title": job['title'],
            "job_id": job_id,
            "level": job['level'],
            "family": job['family'],
            "soc_category": job['soc_category']
        })
    
    # Sort by level then title
    results.sort(key=lambda x: (x['level'], x['title']))
    
    return jsonify({
        "results": results[:limit],
        "total": len(results),
        "filters": {
            "family": family,
            "level": level,
            "min_level": min_level,
            "max_level": max_level
        }
    })

@app.route('/families', methods=['GET'])
def get_families():
    """Get all job families"""
    families = sorted(set(job['family'] for job in job_lookup.values()))
    family_counts = {}
    for family in families:
        count = sum(1 for job in job_lookup.values() if job['family'] == family)
        family_counts[family] = count
    
    return jsonify({
        "families": families,
        "counts": family_counts
    })

@app.route('/levels', methods=['GET'])
def get_levels():
    """Get level information"""
    level_info = {
        0: "Intern/Entry",
        1: "Junior",
        2: "Mid-Level",
        3: "Senior",
        4: "Staff/Manager",
        5: "Senior Manager",
        6: "Director",
        7: "VP",
        8: "Senior VP",
        9: "C-Suite/Executive"
    }
    
    level_counts = {}
    for level in range(10):
        count = sum(1 for job in job_lookup.values() if job['level'] == level)
        if count > 0:
            level_counts[level] = count
    
    return jsonify({
        "levels": level_info,
        "counts": level_counts
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Job Architecture API Server")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /health         - Health check")
    print("  GET  /stats          - System statistics")
    print("  POST /normalize      - Normalize job title")
    print("  POST /career-path    - Get career path")
    print("  POST /search         - Search jobs by criteria")
    print("  GET  /families       - List all job families")
    print("  GET  /levels         - List all levels")
    print("\nStarting server on http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
