"""
Test client for Skill Extractor API

This script demonstrates how to interact with the skill extraction web service.
"""

import requests
import json
from typing import Dict, Any, Optional


class SkillExtractorClient:
    """Client for interacting with the Skill Extractor API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def extract_skills(
        self,
        text: str,
        threshold: float = 0.6,
        top_k: int = 10,
        use_variations: bool = True,
        deduplicate_similar: bool = True,
        dedup_threshold: float = 0.85,
        apply_length_penalty: bool = True,
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract skills from text
        
        Args:
            text: Text to extract skills from
            threshold: Minimum similarity threshold (0-1)
            top_k: Number of top matches to consider per n-gram
            use_variations: Whether to search variations
            deduplicate_similar: Remove similar/redundant skills
            dedup_threshold: Similarity threshold for deduplication
            apply_length_penalty: Apply penalty for partial matches
            max_results: Maximum number of results to return
        
        Returns:
            Dictionary with extracted skills and metadata
        """
        payload = {
            "text": text,
            "threshold": threshold,
            "top_k": top_k,
            "use_variations": use_variations,
            "deduplicate_similar": deduplicate_similar,
            "dedup_threshold": dedup_threshold,
            "apply_length_penalty": apply_length_penalty,
        }
        
        if max_results is not None:
            payload["max_results"] = max_results
        
        response = requests.post(
            f"{self.base_url}/extract",
            json=payload
        )
        response.raise_for_status()
        return response.json()


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example usage of the Skill Extractor API"""
    
    # Create client
    client = SkillExtractorClient(base_url="http://localhost:8000")
    
    # Check health
    print("=" * 80)
    print("HEALTH CHECK")
    print("=" * 80)
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Example text
    sample_text = """
    I have 5 years of experience in Python programming and machine learning. 
    I've worked extensively with TensorFlow and PyTorch for deep learning projects.
    I'm also proficient in SQL databases and cloud platforms like AWS.
    """
    
    # Extract skills
    print("\n" + "=" * 80)
    print("EXTRACTING SKILLS")
    print("=" * 80)
    print(f"Text: {sample_text.strip()}")
    print()
    
    result = client.extract_skills(
        text=sample_text,
        threshold=0.6,
        use_variations=True,
        deduplicate_similar=True,
        apply_length_penalty=True,
        max_results=10
    )
    
    print(f"Total detected: {result['total_detected']}")
    print(f"Processing time: {result['processing_time_ms']:.2f}ms")
    print("\nTop skills:")
    print("-" * 80)
    
    for i, skill in enumerate(result['skills'], 1):
        print(f"\n{i}. {skill['canonical_name']}")
        print(f"   Skill ID: {skill['skill_id']}")
        print(f"   Score: {skill['score']:.3f}")
        print(f"   Matched text: '{skill['matched_text']}'")
        if skill.get('matched_variation'):
            print(f"   Via variation: '{skill['matched_variation']}'")
        if skill.get('raw_similarity'):
            print(f"   Raw similarity: {skill['raw_similarity']:.3f}")
    
    # Get statistics
    print("\n" + "=" * 80)
    print("SERVICE STATISTICS")
    print("=" * 80)
    stats = client.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Example with different parameters
    print("\n" + "=" * 80)
    print("EXTRACTION WITH DIFFERENT PARAMETERS")
    print("=" * 80)
    
    result2 = client.extract_skills(
        text=sample_text,
        threshold=0.7,  # Higher threshold
        use_variations=False,  # Only canonical names
        apply_length_penalty=False,  # No length penalty
        max_results=5
    )
    
    print(f"Total detected: {result2['total_detected']}")
    print(f"Processing time: {result2['processing_time_ms']:.2f}ms")
    print("\nSkills:")
    for skill in result2['skills']:
        print(f"  • {skill['canonical_name']} (ID: {skill['skill_id']}, Score: {skill['score']:.3f})")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the service.")
        print("Make sure the service is running:")
        print("  uvicorn skill_extractor_service:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"ERROR: {e}")
