#!/usr/bin/env python3
"""
FastAPI server for hieroglyph translation pipeline:
Image → Symbol Detection → Classification → Translation/Story
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import shutil
import uuid
import json
from dotenv import load_dotenv

# Import pipeline classes
from hieroglyph_pipeline import HieroglyphPipeline, HieroglyphConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api_server.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hieroglyph Translation API",
    description="API for translating hieroglyph images to stories",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
config = None

# Create base directories
BASE_DIR = Path("output")
IMAGES_DIR = BASE_DIR / "images"
SYMBOLS_DIR = BASE_DIR / "symbols"
TRANSLATIONS_DIR = BASE_DIR / "translations"
JSON_DIR = BASE_DIR / "json"
BY_IMAGE_DIR = BASE_DIR / "by_image"
INDEX_CSV = BASE_DIR / "index.csv"
INDEX_JSONL = BASE_DIR / "index.jsonl"

for dir in [BASE_DIR, IMAGES_DIR, SYMBOLS_DIR, TRANSLATIONS_DIR, JSON_DIR, BY_IMAGE_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

def _ensure_index_headers():
    """Create index files with headers if they don't exist."""
    if not INDEX_CSV.exists():
        header = (
            "timestamp,session_dir,image_filename,image_path,json_path,translation_path,"
            "evaluation_path,symbols_dir,symbols_found,processing_time,error\n"
        )
        INDEX_CSV.write_text(header, encoding="utf-8")
    if not INDEX_JSONL.exists():
        INDEX_JSONL.touch()

def create_session_directory() -> tuple[Path, dict[str, Path]]:
    """Create a unique session directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = str(uuid.uuid4())[:8]
    session_name = f"session_{timestamp}_{session_id}"
    
    # Create session directories
    dirs = {
        "session": BASE_DIR / session_name,
        "images": IMAGES_DIR / session_name,
        "symbols": SYMBOLS_DIR / session_name,
        "translations": TRANSLATIONS_DIR / session_name,
        "json": JSON_DIR / session_name
    }
    
    for dir in dirs.values():
        dir.mkdir(parents=True, exist_ok=True)
    
    return dirs["session"], dirs

def save_results_to_disk(results: dict, session_dirs: dict[str, Path], filename: str):
    """Save processing results to appropriate directories"""
    base_name = Path(filename).stem
    
    # Save JSON results
    json_path = session_dirs["json"] / f"{base_name}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save translation/story if present
    if results.get('story'):
        translation_path = session_dirs["translations"] / f"{base_name}_translation.txt"
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(results['story'])
    
    # Save evaluation if present
    if results.get('evaluation'):
        evaluation_path = session_dirs["translations"] / f"{base_name}_evaluation.json"
        with open(evaluation_path, 'w', encoding='utf-8') as f:
            eval_obj = results['evaluation']
            if isinstance(eval_obj, dict):
                eval_dict = eval_obj
            else:
                try:
                    eval_dict = {
                        "overall_score": eval_obj.overall_score,
                        "criteria_scores": {k.value: v for k, v in eval_obj.criteria_scores.items()},
                        "strengths": eval_obj.strengths,
                        "weaknesses": eval_obj.weaknesses,
                        "suggestions": eval_obj.suggestions,
                        "historical_context": eval_obj.historical_context,
                        "confidence_analysis": eval_obj.confidence_analysis
                    }
                except Exception:
                    eval_dict = {}
            json.dump(eval_dict, f, indent=2, ensure_ascii=False)
    
    # Update results with file paths
    results['file_paths'] = {
        'json_results': str(json_path),
        'translation': str(translation_path) if results.get('story') else None,
        'evaluation': str(evaluation_path) if results.get('evaluation') else None,
        'symbols_dir': str(session_dirs["symbols"]),
        'input_image': str(session_dirs["images"] / filename)
    }

    # Also organize per-image folder under output/by_image/<image_name>/
    image_root = BY_IMAGE_DIR / base_name
    image_root.mkdir(parents=True, exist_ok=True)

    # Write consolidated copies under per-image folder
    # Copy results.json
    consolidated_json = image_root / "results.json"
    with open(consolidated_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Copy translation/evaluation if present
    consolidated_translation = None
    if results.get('story'):
        consolidated_translation = image_root / "translation.txt"
        with open(consolidated_translation, 'w', encoding='utf-8') as f:
            f.write(results['story'])

    consolidated_evaluation = None
    if results.get('evaluation'):
        consolidated_evaluation = image_root / "evaluation.json"
        with open(consolidated_evaluation, 'w', encoding='utf-8') as f:
            eval_obj = results['evaluation']
            try:
                eval_dict = {
                    "overall_score": eval_obj.overall_score,
                    "criteria_scores": {k.value: v for k, v in eval_obj.criteria_scores.items()},
                    "strengths": eval_obj.strengths,
                    "weaknesses": eval_obj.weaknesses,
                    "suggestions": eval_obj.suggestions,
                    "historical_context": eval_obj.historical_context,
                    "confidence_analysis": eval_obj.confidence_analysis,
                }
            except Exception:
                # If already dict-like
                eval_dict = eval_obj
            json.dump(eval_dict, f, indent=2, ensure_ascii=False)

    # Copy symbols into per-image folder
    consolidated_symbols = image_root / "symbols"
    consolidated_symbols.mkdir(exist_ok=True)
    try:
        for item in session_dirs["symbols"].glob("*.png"):
            target = consolidated_symbols / item.name
            try:
                # Copy file content
                with open(item, 'rb') as src, open(target, 'wb') as dst:
                    dst.write(src.read())
            except Exception:
                pass
    except Exception:
        pass

    # Copy input image
    try:
        input_img_src = session_dirs["images"] / filename
        input_img_dst = image_root / filename
        with open(input_img_src, 'rb') as src, open(input_img_dst, 'wb') as dst:
            dst.write(src.read())
    except Exception:
        pass

    # Update results file_paths to reference per-image consolidated locations as primary
    results['file_paths_by_image'] = {
        'root_dir': str(image_root),
        'json_results': str(consolidated_json),
        'translation': str(consolidated_translation) if consolidated_translation else None,
        'evaluation': str(consolidated_evaluation) if consolidated_evaluation else None,
        'symbols_dir': str(consolidated_symbols),
        'input_image': str(image_root / filename),
    }

def update_global_index(results: dict, session_dirs: dict[str, Path], filename: str):
    """Append a row to output/index.csv and a JSONL entry to output/index.jsonl for analysis."""
    _ensure_index_headers()
    from datetime import datetime
    ts = datetime.now().isoformat()
    by_image = results.get('file_paths_by_image', {})
    json_path = by_image.get('json_results') or results.get('file_paths', {}).get('json_results')
    translation_path = by_image.get('translation') or results.get('file_paths', {}).get('translation')
    evaluation_path = by_image.get('evaluation') or results.get('file_paths', {}).get('evaluation')
    row = (
        f"{ts},"
        f"{session_dirs['session']},"
        f"{filename},"
        f"{session_dirs['images'] / filename},"
        f"{json_path or ''},"
        f"{translation_path or ''},"
        f"{evaluation_path or ''},"
        f"{session_dirs['symbols']},"
        f"{results.get('symbols_found', 0)},"
        f"{results.get('processing_time', 0)},"
        f"{(results.get('error') or '').replace(',', ';')}\n"
    )
    with INDEX_CSV.open('a', encoding='utf-8') as f:
        f.write(row)
    # JSONL: compact entry
    entry = {
        'timestamp': ts,
        'session_dir': str(session_dirs['session']),
        'image_filename': filename,
        'image_path': str(session_dirs['images'] / filename),
        'json_path': json_path,
        'translation_path': translation_path,
        'evaluation_path': evaluation_path,
        'symbols_dir': str(session_dirs['symbols']),
        'symbols_found': results.get('symbols_found', 0),
        'processing_time': results.get('processing_time', 0),
        'error': results.get('error'),
    }
    with INDEX_JSONL.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

@app.on_event("startup")
async def startup():
    """Initialize the pipeline on startup"""
    global pipeline, config
    try:
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Load OpenRouter key from RAG/.env or RAG/src/.env
        base_dir = os.path.abspath(os.path.dirname(__file__))
        for env_path in (
            os.path.join(base_dir, 'RAG', '.env'),
            os.path.join(base_dir, 'RAG', 'src', '.env'),
        ):
            if os.path.exists(env_path):
                load_dotenv(env_path)
        
        # Initialize configuration
        config = HieroglyphConfig()
        
        # Check for required model files
        if not os.path.exists(config.SAM_CHECKPOINT_PATH):
            raise FileNotFoundError(f"SAM model not found at {config.SAM_CHECKPOINT_PATH}")
        if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
            raise FileNotFoundError(f"Classifier model not found at {config.CLASSIFIER_MODEL_PATH}")
        
        # Read API key from environment (None disables base LLM story)
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        
        # Initialize pipeline
        pipeline = HieroglyphPipeline(config, api_key=api_key)
        pipeline.setup()
        _ensure_index_headers()
        
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hieroglyph Translation API",
        "status": "running",
        "endpoints": {
            "/translate": "Translate a hieroglyph image",
            "/health": "Check API health",
            "/config": "Get current configuration"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        return {"status": "error", "message": "Pipeline not initialized"}
    return {"status": "healthy", "message": "Pipeline ready"}

@app.post("/translate")
async def translate_hieroglyph(file: UploadFile = File(...)):
    """Translate a hieroglyph image to text"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create session directories
        session_dir, session_dirs = create_session_directory()
        logger.info(f"Created session directory: {session_dir}")
        
        # Save uploaded file
        input_path = session_dirs["images"] / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        results = pipeline.process_image(str(input_path), str(session_dirs["symbols"]))

        # Optional story-only mode: hide classifications/symbols in response
        try:
            from fastapi import Request  # type: ignore
        except Exception:
            Request = None  # pragma: no cover
        # Access query params via request in FastAPI dependency graph
        # Fallback: check a flag set by reverse proxy (not required)
        story_only = False
        try:
            # starlette request is available via app routes, use request scope if injected
            # Since we don't inject Request here, read from the uploaded file's headers as fallback
            story_only = 'story_only=1' in (file.headers.get('x-extra-params', '') if hasattr(file, 'headers') else '')
        except Exception:
            story_only = False

        # Add symbol images as base64 data for web display (skip if story_only)
        if results.get('classifications') and not story_only:
            import base64
            for i, classification in enumerate(results['classifications']):
                if 'path' in classification:
                    try:
                        # Read the symbol image file
                        with open(classification['path'], 'rb') as img_file:
                            img_data = img_file.read()
                            # Convert to base64
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            classification['symbol_image_base64'] = img_base64
                    except Exception as e:
                        logger.warning(f"Could not encode symbol image {classification.get('path', 'unknown')}: {e}")
                        classification['symbol_image_base64'] = None

        if story_only:
            # Remove heavy symbol data from response while keeping files on disk and index updated
            results['classifications'] = []
            if 'translation_metadata' not in results:
                results['translation_metadata'] = {}
            results['translation_metadata']['analysis'] = 'hidden'

        # Serialize evaluation object for API response if needed
        if results.get('evaluation') is not None:
            eval_obj = results['evaluation']
            try:
                eval_dict = {
                    "overall_score": eval_obj.overall_score,
                    "criteria_scores": {k.value: v for k, v in eval_obj.criteria_scores.items()},
                    "strengths": eval_obj.strengths,
                    "weaknesses": eval_obj.weaknesses,
                    "suggestions": eval_obj.suggestions,
                    "historical_context": eval_obj.historical_context,
                    "confidence_analysis": eval_obj.confidence_analysis,
                }
                results['evaluation'] = eval_dict
            except Exception:
                # If already serializable, leave as is
                pass
        
        # Save results to disk
        save_results_to_disk(results, session_dirs, file.filename)
        # Update global index files for analysis
        update_global_index(results, session_dirs, file.filename)
        
        # Add session information to results
        results["session_dir"] = str(session_dir)
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current pipeline configuration"""
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not available")
    
    return {
        "image_size": config.IMAGE_SIZE,
        "sam_model_type": config.SAM_MODEL_TYPE,
        "min_mask_area": config.MIN_MASK_AREA,
        "story_enabled": config.STORY_ENABLED,
        "llm_model": config.LLM_MODEL if hasattr(config, 'LLM_MODEL') else None,
        "output_directories": {
            "base": str(BASE_DIR),
            "images": str(IMAGES_DIR),
            "symbols": str(SYMBOLS_DIR),
            "translations": str(TRANSLATIONS_DIR),
            "json": str(JSON_DIR)
        }
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )