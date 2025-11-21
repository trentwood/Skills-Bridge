#!/usr/bin/env python3
"""
Test script for Job Architecture API
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:5001"

def print_section(title: str):
    """Print a section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def test_health():
    """Test health endpoint"""
    print_section("Health Check")
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(json.dumps(data, indent=2))
    return data['status'] == 'healthy'

def test_stats():
    """Test statistics endpoint"""
    print_section("System Statistics")
    response = requests.get(f"{BASE_URL}/stats")
    data = response.json()
    print(f"Total Jobs: {data['total_jobs']:,}")
    print(f"Searchable Titles: {data['total_searchable_titles']:,}")
    print(f"Job Families: {len(data['families'])}")
    print(f"Organizational Levels: {len(data['levels'])}")

def test_families():
    """Test families endpoint"""
    print_section("Job Families")
    response = requests.get(f"{BASE_URL}/families")
    data = response.json()
    print(f"Total Families: {len(data['families'])}")
    print("\nTop 10 families by job count:")
    sorted_families = sorted(data['counts'].items(), key=lambda x: x[1], reverse=True)
    for family, count in sorted_families[:10]:
        print(f"  {family:30} {count:>6,} jobs")

def test_levels():
    """Test levels endpoint"""
    print_section("Organizational Levels")
    response = requests.get(f"{BASE_URL}/levels")
    data = response.json()
    print("\nLevel Distribution:")
    for level in sorted([int(k) for k in data['counts'].keys()]):
        count = data['counts'][str(level)]
        desc = data['levels'][str(level)]
        print(f"  Level {level} ({desc:20}) {count:>6,} jobs")

def test_normalize():
    """Test job title normalization"""
    print_section("Job Title Normalization")
    
    test_titles = [
        "Software Developer",
        "ML Engineer",
        "Product Manager",
        "VP Engineering",
        "Data Analyst",
        "UX Designer",
        "Sales Representative",
        "Chief Technology Officer",
    ]
    
    for title in test_titles:
        print(f"\n{'─'*70}")
        print(f"Input: '{title}'")
        print(f"{'─'*70}")
        
        response = requests.post(
            f"{BASE_URL}/normalize",
            json={"title": title, "top_k": 3}
        )
        
        if response.status_code == 200:
            results = response.json()
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']}")
                print(f"   Score: {r['similarity_score']:.3f} | "
                      f"Level: {r['level']} | "
                      f"Family: {r['family']} | "
                      f"Type: {r['match_type']}")
        else:
            print(f"   Error: {response.status_code}")

def test_career_path():
    """Test career path endpoint"""
    print_section("Career Path Analysis")
    
    test_cases = [
        ("Software Engineer", "up"),
        ("Engineering Manager", "down"),
        ("Product Manager", "lateral"),
        ("Senior Data Scientist", "up"),
    ]
    
    for title, direction in test_cases:
        print(f"\n{'─'*70}")
        print(f"Career path for '{title}' (direction: {direction})")
        print(f"{'─'*70}")
        
        response = requests.post(
            f"{BASE_URL}/career-path",
            json={"title": title, "direction": direction, "limit": 5}
        )
        
        if response.status_code == 200:
            result = response.json()
            current = result['current_job']
            print(f"\nCurrent: {current['title']}")
            print(f"  Level: {current['level']}, Family: {current['family']}")
            
            print(f"\nPath ({result['total_available']} total options, showing 5):")
            for job in result['path']:
                print(f"  • {job['title']}")
                print(f"    Level: {job['level']}, Family: {job['family']}")
        else:
            print(f"   Error: {response.status_code}")

def test_search():
    """Test search endpoint"""
    print_section("Job Search")
    
    test_searches = [
        {"family": "Engineering", "min_level": 3, "max_level": 4, "limit": 10},
        {"family": "Data", "level": 3, "limit": 10},
        {"level": 9, "limit": 20},  # All executives
        {"min_level": 6, "max_level": 7, "limit": 15},  # Directors and VPs
    ]
    
    for i, search_params in enumerate(test_searches, 1):
        print(f"\n{'─'*70}")
        print(f"Search {i}: {search_params}")
        print(f"{'─'*70}")
        
        response = requests.post(
            f"{BASE_URL}/search",
            json=search_params
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Found {result['total']} jobs, showing {len(result['results'])}")
            
            for job in result['results'][:10]:
                print(f"  • {job['title']}")
                print(f"    Level: {job['level']}, Family: {job['family']}")
        else:
            print(f"   Error: {response.status_code}")

def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*70)
    print(" JOB ARCHITECTURE API - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    try:
        # Test connectivity
        if not test_health():
            print("\n❌ Service is not healthy!")
            return
        
        # Run all tests
        test_stats()
        test_families()
        test_levels()
        test_normalize()
        test_career_path()
        test_search()
        
        print("\n" + "="*70)
        print(" ✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to service.")
        print("Make sure the service is running:")
        print("  python job_architecture_service.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    run_all_tests()
