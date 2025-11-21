"""
Skill Extractor Web Service

A FastAPI-based web service for extracting skills from text using a pre-built taxonomy.

Usage:
    uvicorn skill_extractor_service:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /extract - Extract skills from text
    GET /health - Health check
    GET /stats - Service statistics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ollama for LLM validation
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. LLM validation will not be available.")

# ============================================================================
# Data Models
# ============================================================================

class ExtractionRequest(BaseModel):
    """Request model for skill extraction"""
    text: str = Field(..., description="Text to extract skills from", min_length=1)
    threshold: float = Field(0.6, description="Minimum similarity threshold (0-1)", ge=0, le=1)
    top_k: int = Field(10, description="Number of top matches to consider per n-gram", ge=1, le=50)
    use_variations: bool = Field(True, description="Whether to search variations index")
    deduplicate_similar: bool = Field(True, description="Remove similar/redundant skills")
    dedup_threshold: float = Field(0.85, description="Similarity threshold for deduplication", ge=0, le=1)
    apply_length_penalty: bool = Field(True, description="Apply penalty for partial matches")
    max_results: Optional[int] = Field(None, description="Maximum number of results to return", ge=1)
    # LLM validation parameters
    validate_relevance: bool = Field(True, description="Use LLM to validate skill relevance")
    context: Optional[str] = Field(None, description="Context hint for relevance validation (e.g., 'software engineer')")
    ollama_model: str = Field("llama3.2:3b", description="Ollama model for validation")
    relevance_threshold: float = Field(0.5, description="Minimum relevance score to keep skill", ge=0, le=1)


class SkillResult(BaseModel):
    """Model for a single skill result"""
    skill_id: str = Field(..., description="Unique skill identifier")
    canonical_name: str = Field(..., description="Canonical skill name")
    score: float = Field(..., description="Adjusted similarity score")
    matched_text: str = Field(..., description="Text that matched from input")
    matched_variation: Optional[str] = Field(None, description="Variation that was matched")
    raw_similarity: Optional[float] = Field(None, description="Raw similarity before adjustments")
    relevance_score: Optional[float] = Field(None, description="LLM-assessed relevance to context (0-1)")


class ExtractionResponse(BaseModel):
    """Response model for skill extraction"""
    skills: List[SkillResult]
    total_detected: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    skills_loaded: int
    variations_loaded: int
    ollama_available: bool = Field(False, description="Whether Ollama is available for LLM validation")


class StatsResponse(BaseModel):
    """Service statistics response"""
    total_requests: int
    total_skills_extracted: int
    avg_skills_per_request: float
    model_name: str
    taxonomy_size: int


# ============================================================================
# SkillExtractor Class
# ============================================================================

class SkillExtractor:
    """
    Fast skill extraction using pre-built taxonomy and numpy similarity search
    with improved scoring and deduplication
    """
    
    def __init__(self,
                 taxonomy_path: str = None,
                 variations_path: str = None,
                 canonical_embeddings_path: str = None,
                 variations_embeddings_path: str = None,
                 model_name: str = 'all-MiniLM-L6-v2'):

        # Set default paths relative to project root
        data_dir = Path(__file__).parent.parent.parent / "data" / "skills"
        if taxonomy_path is None:
            taxonomy_path = str(data_dir / 'skill_taxonomy.parquet')
        if variations_path is None:
            variations_path = str(data_dir / 'skill_variations.parquet')
        if canonical_embeddings_path is None:
            canonical_embeddings_path = str(data_dir / 'skill_canonical_embeddings.npy')
        if variations_embeddings_path is None:
            variations_embeddings_path = str(data_dir / 'skill_variations_embeddings.npy')
        
        logger.info("Loading skill extractor...")
        
        # Check if files exist
        self._check_files_exist([
            taxonomy_path,
            variations_path,
            canonical_embeddings_path,
            variations_embeddings_path
        ])
        
        # Load taxonomy
        self.skills_df = pd.read_parquet(taxonomy_path)
        self.variations_df = pd.read_parquet(variations_path)
        
        # Load embeddings (numpy arrays)
        self.canonical_embeddings = np.load(canonical_embeddings_path)
        self.variations_embeddings = np.load(variations_embeddings_path)
        
        # Load embedding model
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Statistics
        self.total_requests = 0
        self.total_skills_extracted = 0
        
        logger.info(f"✓ Loaded {len(self.skills_df)} skills")
        logger.info(f"✓ Loaded {len(self.variations_df)} variations")
        logger.info(f"✓ Model: {model_name}")
        logger.info("✓ Ready for extraction")
    
    def _check_files_exist(self, filepaths: List[str]):
        """Check if required files exist"""
        missing_files = []
        for filepath in filepaths:
            if not Path(filepath).exists():
                missing_files.append(filepath)
        
        if missing_files:
            raise FileNotFoundError(
                f"Required files not found: {', '.join(missing_files)}\n"
                f"Please ensure the taxonomy and embedding files are in the same directory."
            )
    
    def extract_from_text(self, 
                         text: str, 
                         threshold: float = 0.6,
                         top_k: int = 10, 
                         use_variations: bool = True,
                         deduplicate_similar: bool = True, 
                         dedup_threshold: float = 0.85,
                         apply_length_penalty: bool = True,
                         max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract skills from text using numpy-based similarity search
        
        Args:
            text: Input text to extract skills from
            threshold: Minimum similarity threshold (0-1)
            top_k: Number of top matches to consider per n-gram
            use_variations: Whether to search variations or just canonical names
            deduplicate_similar: Remove skills that are very similar to each other
            dedup_threshold: Similarity threshold for considering skills duplicates
            apply_length_penalty: Penalize matches where ngram is much shorter than skill
            max_results: Maximum number of results to return
        
        Returns:
            List of detected skills with metadata
        """
        # Update statistics
        self.total_requests += 1
        
        # Generate n-grams from text
        ngrams = self._generate_ngrams(text, max_n=5)
        
        if not ngrams:
            return []
        
        # Encode n-grams
        ngram_embeddings = self.model.encode(ngrams, convert_to_numpy=True)
        ngram_embeddings = ngram_embeddings / np.linalg.norm(
            ngram_embeddings, axis=1, keepdims=True
        )
        
        # Search for matches
        detected_skills = {}
        
        for i, ngram in enumerate(ngrams):
            query = ngram_embeddings[i:i+1]
            
            if use_variations:
                # Compute cosine similarity with all variations (dot product)
                similarities = np.dot(self.variations_embeddings, query.T).flatten()
                
                # Get top k indices
                if len(similarities) > top_k:
                    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
                else:
                    top_indices = np.argsort(similarities)[::-1]
                
                for idx in top_indices:
                    dist = similarities[idx]
                    if dist >= threshold:
                        match = self.variations_df.iloc[idx]
                        skill_id = match['skill_id']
                        canonical = match['canonical_name']
                        matched_variation = match['variation']
                        
                        # Apply length penalty
                        if apply_length_penalty:
                            adjusted_score = self._apply_length_penalty(
                                dist, ngram, canonical
                            )
                        else:
                            adjusted_score = dist
                        
                        # Only update if this is a better match
                        if skill_id not in detected_skills or adjusted_score > detected_skills[skill_id]['score']:
                            detected_skills[skill_id] = {
                                'skill_id': skill_id,
                                'canonical_name': canonical,
                                'matched_text': ngram,
                                'matched_variation': matched_variation,
                                'score': float(adjusted_score),
                                'raw_similarity': float(dist)
                            }
            else:
                # Search canonical embeddings
                similarities = np.dot(self.canonical_embeddings, query.T).flatten()
                
                if len(similarities) > top_k:
                    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
                else:
                    top_indices = np.argsort(similarities)[::-1]
                
                for idx in top_indices:
                    dist = similarities[idx]
                    if dist >= threshold:
                        match = self.skills_df.iloc[idx]
                        skill_id = match['skill_id']
                        canonical = match['canonical_name']
                        
                        # Apply length penalty
                        if apply_length_penalty:
                            adjusted_score = self._apply_length_penalty(
                                dist, ngram, canonical
                            )
                        else:
                            adjusted_score = dist
                        
                        if skill_id not in detected_skills or adjusted_score > detected_skills[skill_id]['score']:
                            detected_skills[skill_id] = {
                                'skill_id': skill_id,
                                'canonical_name': canonical,
                                'matched_text': ngram,
                                'score': float(adjusted_score),
                                'raw_similarity': float(dist)
                            }
        
        # Sort by score
        results = sorted(detected_skills.values(), key=lambda x: x['score'], reverse=True)
        
        # Deduplicate similar skills
        if deduplicate_similar and len(results) > 1:
            results = self._deduplicate_similar_skills(results, dedup_threshold)
        
        # Limit results if max_results specified
        if max_results is not None and len(results) > max_results:
            results = results[:max_results]
        
        # Update statistics
        self.total_skills_extracted += len(results)
        
        return results
    
    def _apply_length_penalty(self, similarity: float, ngram: str, canonical_skill: str) -> float:
        """
        Apply penalty when matched n-gram is much shorter than the skill name
        This prevents partial matches from scoring too high
        """
        ngram_len = len(ngram.split())
        skill_len = len(canonical_skill.split())
        
        if ngram_len < skill_len:
            # Penalize based on length difference
            length_penalty = ngram_len / skill_len
            # Apply penalty with a minimum floor to avoid over-penalization
            penalty_factor = max(0.5, length_penalty)
            adjusted_score = similarity * penalty_factor
        else:
            adjusted_score = similarity
        
        return adjusted_score
    
    def _deduplicate_similar_skills(self, results: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Remove redundant skills that are very similar to higher-scoring skills
        """
        if len(results) <= 1:
            return results
        
        # Get embeddings for all detected skills
        skill_names = [r['canonical_name'] for r in results]
        skill_embeddings = self.model.encode(skill_names, convert_to_numpy=True)
        skill_embeddings = skill_embeddings / np.linalg.norm(skill_embeddings, axis=1, keepdims=True)
        
        # Keep track of which skills to keep
        keep_indices = []
        
        for i in range(len(results)):
            # Always keep the first (highest scoring)
            if i == 0:
                keep_indices.append(i)
                continue
            
            # Check similarity with all higher-scoring kept skills
            should_keep = True
            for j in keep_indices:
                similarity = np.dot(skill_embeddings[i], skill_embeddings[j])
                if similarity > threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep_indices.append(i)
        
        return [results[i] for i in keep_indices]
    
    def _generate_ngrams(self, text: str, max_n: int = 5) -> List[str]:
        """
        Generate n-grams from text (1 to max_n words)
        """
        # Clean and tokenize
        text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
        tokens = text.lower().split()
        
        ngrams = []
        for n in range(1, min(max_n + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        
        return ngrams
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        avg_skills = (
            self.total_skills_extracted / self.total_requests
            if self.total_requests > 0 else 0
        )

        return {
            'total_requests': self.total_requests,
            'total_skills_extracted': self.total_skills_extracted,
            'avg_skills_per_request': round(avg_skills, 2),
            'model_name': self.model_name,
            'taxonomy_size': len(self.skills_df)
        }

    def validate_relevance_with_llm(
        self,
        text: str,
        skills: List[Dict[str, Any]],
        context: Optional[str] = None,
        model: str = "llama3.2:3b",
        relevance_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Use a local LLM via Ollama to validate skill relevance to the text context.

        Args:
            text: Original text that was analyzed
            skills: List of extracted skills
            context: Optional context hint (e.g., "software engineer job posting")
            model: Ollama model to use
            relevance_threshold: Minimum relevance score to keep skill

        Returns:
            Filtered list of skills with relevance_score added
        """
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available, skipping LLM validation")
            return skills

        if not skills:
            return skills

        # Limit to top 30 skills for efficiency (reduced from 50 to prevent truncation)
        skills_to_validate = skills[:30]
        skill_names = [s['canonical_name'] for s in skills_to_validate]

        # Build context description
        if context:
            context_desc = f"about {context}"
        else:
            context_desc = ""

        # Create prompt for batch validation
        skills_list = "\n".join([f"- {name}" for name in skill_names])

        prompt = f"""Rate each skill's relevance to this text{context_desc}:

{text[:1500]}

Rate from 0.0 (not relevant/incidental) to 1.0 (core requirement).

Skills:
{skills_list}

Return ONLY JSON with skill names as keys and scores as values.
Example: {{"Python": 0.95, "Benefits": 0.1}}

JSON:"""

        try:
            # Call Ollama
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0,  # Zero temperature for deterministic results
                    "num_predict": 3500,  # Increased to handle skill ratings
                }
            )

            response_text = response['response'].strip()

            # Try to parse JSON from response
            # Find JSON object - handle potential nested content
            try:
                # First try to find JSON starting from first {
                start_idx = response_text.find('{')
                if start_idx != -1:
                    # Find matching closing brace
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(response_text[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    json_str = response_text[start_idx:end_idx]
                    relevance_scores = json.loads(json_str)
                else:
                    logger.warning(f"No JSON found in LLM response: {response_text[:200]}")
                    return skills
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON from LLM response: {e}")
                return skills

            # Build lookup dict with normalized keys for better matching
            normalized_scores = {}
            for key, score in relevance_scores.items():
                normalized_scores[key.lower().strip()] = float(score)

            # Apply relevance scores and filter
            validated_skills = []
            unmatched_count = 0
            for skill in skills_to_validate:
                skill_name = skill['canonical_name']
                normalized_name = skill_name.lower().strip()

                # Try exact match first
                if normalized_name in normalized_scores:
                    relevance = normalized_scores[normalized_name]
                else:
                    # Try partial matching for hyphenated/spaced variants
                    relevance = None
                    for key, score in normalized_scores.items():
                        # Check if key contains skill name or vice versa
                        if key in normalized_name or normalized_name in key:
                            relevance = score
                            break
                        # Check without hyphens/underscores
                        key_clean = key.replace('-', ' ').replace('_', ' ')
                        name_clean = normalized_name.replace('-', ' ').replace('_', ' ')
                        if key_clean == name_clean:
                            relevance = score
                            break

                    if relevance is None:
                        # Not found - assign low score for incidental mentions
                        relevance = 0.1
                        unmatched_count += 1

                skill['relevance_score'] = relevance

                # Filter by threshold
                if relevance >= relevance_threshold:
                    validated_skills.append(skill)

            if unmatched_count > 0:
                logger.info(f"LLM did not rate {unmatched_count} skills (assigned 0.1)")

            # Sort by relevance score (descending), then by original score
            validated_skills.sort(key=lambda x: (x.get('relevance_score', 0), x['score']), reverse=True)

            logger.info(f"LLM validation: {len(skills_to_validate)} -> {len(validated_skills)} skills")
            return validated_skills

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            # Return original skills if validation fails
            return skills


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Skill Extractor API",
    description="Extract skills from text using semantic similarity search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global extractor instance
extractor: Optional[SkillExtractor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the skill extractor on startup"""
    global extractor
    try:
        extractor = SkillExtractor()
        logger.info("Skill extractor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize skill extractor: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Skill Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "extract": "/extract",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if extractor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        skills_loaded=len(extractor.skills_df),
        variations_loaded=len(extractor.variations_df),
        ollama_available=OLLAMA_AVAILABLE
    )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """Get service statistics"""
    if extractor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    stats = extractor.get_stats()
    return StatsResponse(**stats)


@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_skills(request: ExtractionRequest):
    """
    Extract skills from text
    
    Args:
        request: ExtractionRequest with text and parameters
    
    Returns:
        ExtractionResponse with detected skills
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        import time
        start_time = time.time()

        # Extract skills
        results = extractor.extract_from_text(
            text=request.text,
            threshold=request.threshold,
            top_k=request.top_k,
            use_variations=request.use_variations,
            deduplicate_similar=request.deduplicate_similar,
            dedup_threshold=request.dedup_threshold,
            apply_length_penalty=request.apply_length_penalty,
            max_results=request.max_results
        )

        # Optionally validate relevance with LLM
        if request.validate_relevance and results:
            results = extractor.validate_relevance_with_llm(
                text=request.text,
                skills=results,
                context=request.context,
                model=request.ollama_model,
                relevance_threshold=request.relevance_threshold
            )

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Convert to response model
        skills = [SkillResult(**result) for result in results]

        return ExtractionResponse(
            skills=skills,
            total_detected=len(skills),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting skills: {str(e)}")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "skill_extractor_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
