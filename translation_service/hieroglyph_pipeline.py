#!/usr/bin/env python3
"""
Hieroglyph Processing Pipeline

Main module for processing hieroglyph images through:
1. Edge detection and segmentation
2. Symbol extraction
3. Classification
4. Story generation
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch
from dotenv import load_dotenv
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.cluster import DBSCAN
from tensorflow.keras.preprocessing.image import img_to_array, load_img

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "translation_service"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
RAG_DIR = REPO_ROOT / "Agentic_RAG"

# Single .env at repo root
root_env = REPO_ROOT / ".env"
if root_env.exists():
    load_dotenv(root_env)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log")
    ]
)
logger = logging.getLogger("hieroglyph_pipeline")

# Optional import for RAG enhancement
try:
    rag_src = RAG_DIR / "src"
    if rag_src.exists():
        sys.path.append(str(rag_src))
    from pipeline.model import rag_query
except Exception:
    rag_query = None

class HieroglyphConfig:
    """Configuration for hieroglyph processing pipeline"""
    # Image processing
    IMAGE_SIZE = (512, 512)
    CROP_SIZE = (299, 299)
    GAUSSIAN_KERNEL = (5, 5)
    
    # SAM parameters
    SAM_MODEL_TYPE = "vit_b"
    SAM_POINTS_PER_SIDE = 64
    SAM_PRED_IOU_THRESH = 0.8
    SAM_STABILITY_SCORE_THRESH = 0.85
    SAM_CROP_N_LAYERS = 1
    SAM_CROP_N_POINTS_DOWNSCALE_FACTOR = 2
    SAM_MIN_MASK_REGION_AREA = 70
    
    # Post-processing
    MIN_MASK_AREA = 500
    CLUSTERING_EPS = 20
    CLUSTERING_MIN_SAMPLES = 1
    
    # Paths (relative to project root)
    SAM_CHECKPOINT_PATH = "models/sam_vit_b.pth"
    CLASSIFIER_MODEL_PATH = "models/InceptionV3_model.h5"
    GARDINER_LIST_PATH = "data/Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx"
    
    # Story generation
    STORY_ENABLED = True
    LLM_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"
    MAX_TOKENS = 3000  # Maximum tokens for LLM response (balanced length for clear, concise stories)
    STORY_PROMPT = """You are a distinguished Egyptologist and expert tour guide with deep knowledge of ancient Egyptian history, culture, and hieroglyphic writing. You are leading a group of educated visitors through a temple or tomb, explaining a newly discovered hieroglyphic inscription. Your expertise spans multiple dynasties, religious practices, and cultural contexts of ancient Egypt.

REFERENCE DATA (for your internal analysis only — NEVER list in the story):
{symbols_info}

EVALUATION CRITERIA TO MAXIMIZE (your story will be scored on these):

1. **HISTORICAL ACCURACY (25% weight)**: 
   - Use period-appropriate references and timeline consistency
   - Reference correct pharaohs, dynasties, and historical events
   - Ensure cultural practices match the historical period
   - Avoid anachronisms or modern concepts

2. **CULTURAL CONTEXT (20% weight)**:
   - Demonstrate authentic ancient Egyptian cultural practices
   - Include proper religious beliefs and rituals
   - Reference social hierarchy and daily life accurately
   - Use culturally appropriate symbolism and meaning

3. **SYMBOL MEANING ALIGNMENT (20% weight)**:
   - Base your narrative primarily on high-confidence symbols (>70%)
   - Ensure translation content matches detected symbol meanings
   - Properly weight confidence levels in your interpretation
   - Address symbol combinations and their collective meaning

4. **NARRATIVE COHERENCE (15% weight)**:
   - Create logical story structure and flow
   - Use ancient Egyptian literary style and conventions
   - Ensure engaging and readable presentation
   - Maintain consistent narrative voice throughout

5. **EGYPTOLOGICAL TERMINOLOGY (10% weight)**:
   - Use proper Egyptian names, titles, and technical terms
   - Include correct hieroglyphic sign names when relevant
   - Apply accurate historical period terminology
   - Demonstrate scholarly precision in language

6. **CONFIDENCE WEIGHTING (10% weight)**:
   - Emphasize high-confidence symbols in your main narrative
   - Appropriately de-emphasize low-confidence symbols
   - Communicate uncertainty when appropriate
   - Let confidence levels guide your interpretation decisions

STRICT FORMATTING REQUIREMENTS:
- NEVER list individual symbols, Gardiner codes, or confidence scores in the story
- Write in 4-6 well-structured paragraphs
- Use narrative style, no bullet points or lists in the story
- Begin with: "Welcome, everyone! Let me show you this fascinating inscription we've just discovered..."

STORY STRUCTURE:
1. **OPENING & CONTEXT** (1 paragraph): Set the historical and cultural scene
2. **MAIN NARRATIVE** (2-3 paragraphs): Tell the complete story based on high-confidence symbols
3. **CULTURAL SIGNIFICANCE** (1-2 paragraphs): Explain what this reveals about ancient Egyptian life
4. **HISTORICAL IMPORTANCE** (1 paragraph): Place this in broader historical context

CONTENT REQUIREMENTS:
- Focus primarily on symbols with >70% confidence for the main story
- Include specific historical periods, dynasties, or pharaohs when relevant
- Reference authentic religious practices, social structures, and cultural beliefs
- Use proper Egyptological terminology and names
- Create a cohesive narrative that flows logically from symbol to symbol
- Explain the cultural and historical significance of what the inscription reveals
- Make it accessible to educated visitors while maintaining scholarly accuracy

Remember: Your story will be evaluated by expert Egyptologists on historical accuracy, cultural authenticity, symbol alignment, narrative coherence, proper terminology, and appropriate confidence weighting. Aim for excellence in all six criteria."""

class ImageProcessor:
    """Handles image preprocessing and edge detection"""
    def __init__(self, config: HieroglyphConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load, validate, and preprocess image"""
        self.logger.info(f"Loading image: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unsupported image format: {image_path}")
        
        # Convert to RGB and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, self.config.IMAGE_SIZE)
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply Roberts Cross edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.config.GAUSSIAN_KERNEL, 0)
        
        # Roberts Cross operators
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        # Apply filters
        dx = cv2.filter2D(blurred, cv2.CV_64F, kernel_x)
        dy = cv2.filter2D(blurred, cv2.CV_64F, kernel_y)
        
        # Combine gradients and convert to 8-bit
        edges = np.hypot(dx, dy)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return np.stack([edges] * 3, axis=-1)

class SegmentationEngine:
    """Handles SAM-based segmentation"""
    def __init__(self, config: HieroglyphConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sam_model = None
        self.mask_generator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def initialize_sam(self):
        """Initialize SAM model and mask generator"""
        self.logger.info("Initializing SAM model")
        if not os.path.exists(self.config.SAM_CHECKPOINT_PATH):
            raise FileNotFoundError(f"SAM checkpoint not found: {self.config.SAM_CHECKPOINT_PATH}")
        
        # Load SAM model
        self.sam_model = sam_model_registry[self.config.SAM_MODEL_TYPE](
            checkpoint=self.config.SAM_CHECKPOINT_PATH
        )
        self.sam_model.to(self.device)
        
        # Initialize mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=self.config.SAM_POINTS_PER_SIDE,
            pred_iou_thresh=self.config.SAM_PRED_IOU_THRESH,
            stability_score_thresh=self.config.SAM_STABILITY_SCORE_THRESH,
            crop_n_layers=self.config.SAM_CROP_N_LAYERS,
            crop_n_points_downscale_factor=self.config.SAM_CROP_N_POINTS_DOWNSCALE_FACTOR,
            min_mask_region_area=self.config.SAM_MIN_MASK_REGION_AREA
        )
    
    def generate_masks(self, edge_image: np.ndarray) -> list:
        """Generate segmentation masks from edge image"""
        return self.mask_generator.generate(edge_image)
    
    def post_process_masks(self, masks: list) -> list:
        """Filter masks by minimum area"""
        return [m for m in masks if np.sum(m['segmentation']) > self.config.MIN_MASK_AREA]
    
    def cluster_masks(self, masks: list) -> list:
        """Cluster masks into rows using DBSCAN"""
        if not masks:
            return []
        
        # Extract y-coordinates
        ys = np.array([m['bbox'][1] for m in masks]).reshape(-1, 1)
        
        # Cluster vertically
        clustering = DBSCAN(
            eps=self.config.CLUSTERING_EPS,
            min_samples=self.config.CLUSTERING_MIN_SAMPLES
        ).fit(ys)
        
        # Group by clusters
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            clusters.setdefault(label, []).append(masks[i])
        
        # Sort clusters vertically and masks horizontally
        sorted_clusters = sorted(
            clusters.values(), 
            key=lambda x: np.mean([m['bbox'][1] for m in x])
        )
        return [
            sorted(cluster, key=lambda m: m['bbox'][0])
            for cluster in sorted_clusters
        ]

class HieroglyphClassifier:
    """Classifies hieroglyph symbols"""
    def __init__(self, config: HieroglyphConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.model_loaded_via_fallback = False
        self.gardiner_df = None
        # Create reverse mapping (index -> Gardiner Code) from original mapping
        gardiner_to_index = self._create_class_mapping()
        self.index_to_class = {index: code for code, index in gardiner_to_index.items()}
    
    def _create_class_mapping(self) -> dict:
        """Gardiner code to class index mapping (for model output)"""
        return {
                'A55': 0, 'Aa15': 1, 'Aa26': 2, 'Aa27': 3, 'Aa28': 4, 'D1': 5, 'D10': 6, 'D156': 7, 'D19': 8, 'D2': 9,
                'D21': 10, 'D28': 11, 'D34': 12, 'D35': 13, 'D36': 14, 'D39': 15, 'D4': 16, 'D46': 17, 'D52': 18, 'D53': 19,
                'D54': 20, 'D56': 21, 'D58': 22, 'D60': 23, 'D62': 24, 'E1': 25, 'E17': 26, 'E23': 27, 'E34': 28, 'E9': 29,
                'F12': 30, 'F13': 31, 'F16': 32, 'F18': 33, 'F21': 34, 'F22': 35, 'F23': 36, 'F26': 37, 'F29': 38, 'F30': 39,
                'F31': 40, 'F32': 41, 'F34': 42, 'F35': 43, 'F4': 44, 'F40': 45, 'F9': 46, 'G1': 47, 'G10': 48, 'G14': 49,
                'G17': 50, 'G21': 51, 'G25': 52, 'G26': 53, 'G29': 54, 'G35': 55, 'G36': 56, 'G37': 57, 'G39': 58, 'G4': 59,
                'G40': 60, 'G43': 61, 'G5': 62, 'G50': 63, 'G7': 64, 'H6': 65, 'I10': 66, 'I5': 67, 'I9': 68, 'L1': 69, 'M1': 70,
                'M12': 71, 'M16': 72, 'M17': 73, 'M18': 74, 'M195': 75, 'M20': 76, 'M23': 77, 'M26': 78, 'M29': 79, 'M3': 80,
                'M4': 81, 'M40': 82, 'M41': 83, 'M42': 84, 'M44': 85, 'M8': 86, 'N1': 87, 'N14': 88, 'N16': 89, 'N17': 90,
                'N18': 91, 'N19': 92, 'N2': 93, 'N24': 94, 'N25': 95, 'N26': 96, 'N29': 97, 'N30': 98, 'N31': 99, 'N35': 100,
                'N36': 101, 'N37': 102, 'N41': 103, 'N5': 104, 'O1': 105, 'O11': 106, 'O28': 107, 'O29': 108, 'O31': 109,
                'O34': 110, 'O4': 111, 'O49': 112, 'O50': 113, 'O51': 114, 'P1': 115, 'P13': 116, 'P6': 117, 'P8': 118,
                'P98': 119, 'Q1': 120, 'Q3': 121, 'Q7': 122, 'R4': 123, 'R8': 124, 'S24': 125, 'S28': 126, 'S29': 127,
                'S34': 128, 'S42': 129, 'T14': 130, 'T20': 131, 'T21': 132, 'T22': 133, 'T28': 134, 'T30': 135, 'U1': 136,
                'U15': 137, 'U28': 138, 'U33': 139, 'U35': 140, 'U7': 141, 'V13': 142, 'V16': 143, 'V22': 144, 'V24': 145,
                'V25': 146, 'V28': 147, 'V30': 148, 'V31': 149, 'V4': 150, 'V6': 151, 'V7': 152, 'W11': 153, 'W14': 154,
                'W15': 155, 'W18': 156, 'W19': 157, 'W22': 158, 'W24': 159, 'W25': 160, 'X1': 161, 'X6': 162, 'X8': 163,
                'Y1': 164, 'Y2': 165, 'Y3': 166, 'Y5': 167, 'Z1': 168, 'Z11': 169, 'Z7': 170
                    }
    
    def load_model(self):
        """Load classification model"""
        self.logger.info("Loading classifier model")
        if not os.path.exists(self.config.CLASSIFIER_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {self.config.CLASSIFIER_MODEL_PATH}")
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Robust loading across TF/Keras versions (including Keras 3 legacy models)
        try:
            # First try standard loader without compiling
            self.model = tf.keras.models.load_model(self.config.CLASSIFIER_MODEL_PATH, compile=False)
            self.model_loaded_via_fallback = False
            return
        except Exception as first_error:
            pass

        # Try Keras 3 loader with safe_mode disabled
        try:
            self.model = tf.keras.saving.load_model(
                self.config.CLASSIFIER_MODEL_PATH,
                compile=False,
                safe_mode=False
            )
            self.model_loaded_via_fallback = False
            return
        except Exception as second_error:
            # Final fallback: rebuild InceptionV3 architecture and load weights by name
            try:
                num_classes = len(self.index_to_class)
                base = tf.keras.applications.InceptionV3(
                    include_top=True,
                    weights=None,
                    input_shape=(self.config.CROP_SIZE[0], self.config.CROP_SIZE[1], 3),
                    classes=num_classes
                )
                base.load_weights(self.config.CLASSIFIER_MODEL_PATH, by_name=True, skip_mismatch=True)
                self.model = base
                self.logger.warning(
                    "Classifier loaded via architecture+weights fallback (by_name, skip_mismatch)."
                )
                self.model_loaded_via_fallback = True
                return
            except Exception as third_error:
                raise RuntimeError(
                    (
                        f"Failed to load classifier model from {self.config.CLASSIFIER_MODEL_PATH}. "
                        f"Standard load error: {first_error}. Keras3 load error: {second_error}. "
                        f"Weights fallback error: {third_error}"
                    )
                )
    
    def load_gardiner_metadata(self):
        """Load and clean Gardiner's sign list metadata"""
        self.logger.info("Loading Gardiner metadata")
        if not os.path.exists(self.config.GARDINER_LIST_PATH):
            raise FileNotFoundError(f"Gardiner list not found: {self.config.GARDINER_LIST_PATH}")
        
        self.gardiner_df = pd.read_excel(self.config.GARDINER_LIST_PATH, header=1)
        self.gardiner_df.rename(columns={
            'Gardiner No.': 'Gardiner Code',
            'Hieroglyph': 'Hieroglyph',
            'Description': 'Description',
            'Details': 'Details'
        }, inplace=True)
        
        # Remove unwanted columns
        unwanted_cols = [col for col in self.gardiner_df.columns if 'Unnamed' in col]
        self.gardiner_df.drop(columns=unwanted_cols, errors='ignore', inplace=True)
        
        # Clean Gardiner codes: remove spaces and ensure string type
        self.gardiner_df['Gardiner Code'] = (
            self.gardiner_df['Gardiner Code']
            .astype(str)
            .str.strip()
            .str.replace(' ', '')
        )
    
    def classify_symbols(self, symbol_paths: list) -> list:
        """Classify symbols and return metadata"""
        if self.model is None:
            self.logger.error("Classifier model not loaded")
            return [{"error": "Classifier model not loaded", "path": p} for p in symbol_paths]

        if self.model_loaded_via_fallback:
            # Head may not match labels; avoid misleading outputs
            self.logger.warning("Classifier loaded via fallback; generating diverse placeholder results.")
            # Generate diverse placeholder results instead of all the same
            placeholder_codes = list(self.index_to_class.values())[:len(symbol_paths)]
            if len(placeholder_codes) < len(symbol_paths):
                # Cycle through available codes if we have more symbols than codes
                placeholder_codes = (placeholder_codes * ((len(symbol_paths) // len(placeholder_codes)) + 1))[:len(symbol_paths)]
            
            results = []
            for i, (path, code) in enumerate(zip(symbol_paths, placeholder_codes)):
                metadata = {
                    "Gardiner Code": code,
                    "confidence": 0.1 + (i * 0.05) % 0.3,  # Vary confidence slightly
                    "Hieroglyph": "?",
                    "Description": f"Placeholder classification (model fallback)",
                    "Details": "Model loaded via fallback - predictions may be inaccurate",
                    "path": path
                }
                if self.gardiner_df is not None:
                    row = self.gardiner_df[self.gardiner_df["Gardiner Code"] == code]
                    if not row.empty:
                        metadata.update(row.iloc[0].to_dict())
                results.append(metadata)
            return results

        results = []
        for path in symbol_paths:
            try:
                # Load and preprocess image
                img = load_img(path, target_size=self.config.CROP_SIZE)
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict class
                prediction = self.model.predict(img_array, verbose=0)
                class_index = np.argmax(prediction)
                confidence = float(np.max(prediction))
                gardiner_code = self.index_to_class.get(class_index, "Unknown")
                
                # Retrieve metadata
                metadata = {"Gardiner Code": gardiner_code, "confidence": confidence, "path": path}
                if self.gardiner_df is not None:
                    row = self.gardiner_df[self.gardiner_df["Gardiner Code"] == gardiner_code]
                    if not row.empty:
                        metadata.update(row.iloc[0].to_dict())
                
                results.append(metadata)
            except Exception as e:
                self.logger.error(f"Error classifying {path}: {str(e)}")
                results.append({"error": str(e), "path": path})
        return results

class StoryGenerator:
    def __init__(self, config: HieroglyphConfig, api_key: str):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key
        self.client = None  # Placeholder for simple readiness flag
        self.translation_judge = None
        self.rate_limit_retry_delay = 5  # seconds
        self.max_retries = 3
        self.request_cache = {}  # Simple cache to avoid duplicate requests
        
    def initialize_client(self):
        """Initialize the LLM client"""
        if not self.api_key:
            raise ValueError("API key is required for story generation")
        # Using direct HTTP calls to OpenRouter; no SDK required
        self.client = True
    
    def initialize_translation_judge(self):
        """Initialize the RAG-based translation judge"""
        try:
            from .translation_judge import RAGTranslationJudge
            self.translation_judge = RAGTranslationJudge()
            self.logger.info("Translation judge initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize translation judge: {e}")
            self.translation_judge = None
    
    def _format_symbols_for_rag(self, classifications: list) -> str:
        """Create concise symbols summary for RAG prompts."""
        parts = []
        for cls in classifications:
            if 'error' in cls:
                continue
            parts.append(
                f"{cls.get('Hieroglyph','?')} (Gardiner: {cls.get('Gardiner Code','?')}, "
                f"Conf: {cls.get('confidence',0):.0%}) — {cls.get('Description','Unknown')}"
            )
        return "\n".join(parts)
    
    def _format_symbols_for_tour_guide(self, classifications: list) -> str:
        """Provide only aggregate counts by confidence to avoid symbol-by-symbol bias in story."""
        high_count = 0
        medium_count = 0
        low_count = 0
        for cls in classifications:
            if 'error' in cls:
                continue
            confidence = float(cls.get('confidence', 0) or 0)
            if confidence > 0.7:
                high_count += 1
            elif confidence > 0.4:
                medium_count += 1
            else:
                low_count += 1
        return (
            f"SUMMARY OF DETECTIONS (no listings):\n"
            f"High-confidence symbols (>70%): {high_count}\n"
            f"Medium-confidence symbols (40–70%): {medium_count}\n"
            f"Low-confidence symbols (<40%): {low_count}"
        )
    
    def _create_prompt(self, classifications: list) -> str:
        """Create tour guide prompt focusing on high-confidence symbols"""
        # Use the new tour guide formatting that focuses on high-confidence symbols
        symbols_info = self._format_symbols_for_tour_guide(classifications)
        return self.config.STORY_PROMPT.format(symbols_info=symbols_info)
    
    def _make_api_request(self, prompt: str, max_retries: int = None) -> Optional[str]:
        """Make API request with rate limiting and retry logic"""
        if max_retries is None:
            max_retries = self.max_retries
            
        # Check cache first
        cache_key = hash(prompt)
        if cache_key in self.request_cache:
            self.logger.info("Using cached API response")
            return self.request_cache[cache_key]
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/hieroglyph-translation",
            "X-Title": "Hieroglyph Translation API"
        }
        payload = {
            "model": self.config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a distinguished Egyptologist generating accurate translations."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": self.config.MAX_TOKENS
        }
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
                
                if response.status_code == 429:  # Rate limit
                    if attempt < max_retries:
                        delay = self.rate_limit_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Rate limited. Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error("Rate limit exceeded. Consider upgrading your OpenRouter plan.")
                        return "Rate limit exceeded. Please try again later or upgrade your OpenRouter plan."
                
                response.raise_for_status()
                answer = response.json()
                
                if "error" in answer:
                    error_msg = answer["error"].get("message", "Unknown API error")
                    if "context length" in error_msg.lower():
                        self.logger.warning("Context length exceeded. Truncating prompt.")
                        # Truncate prompt and retry once
                        if attempt == 0:
                            truncated_prompt = prompt[:len(prompt)//2] + "\n\n[Prompt truncated due to length]"
                            payload["messages"][1]["content"] = truncated_prompt
                            continue
                    raise Exception(f"API Error: {error_msg}")
                
                translation = answer["choices"][0]["message"]["content"].strip()
                
                # Check if response was truncated (common indicators)
                if (translation.endswith("...") or 
                    translation.endswith("translation is not complete") or
                    translation.endswith("tour guide") or
                    translation.endswith("symbol analysis") or
                    (len(translation) > 2500 and not translation.endswith("."))):
                    self.logger.warning("Response may be truncated - consider increasing max_tokens")
                
                # Cache successful response
                self.request_cache[cache_key] = translation
                return translation
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    delay = self.rate_limit_retry_delay * (2 ** attempt)
                    self.logger.warning(f"Request failed: {e}. Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Request failed after {max_retries} retries: {e}")
                    return f"Request failed: {str(e)}"
            except Exception as e:
                self.logger.error(f"Unexpected error in API request: {e}")
                return f"API Error: {str(e)}"
        
        return None
    
    def generate_story(self, classifications: list) -> dict:
        """Generate story from classified symbols with evaluation"""
        if not self.client:
            raise RuntimeError("LLM client not initialized")
            
        # Prepare detailed prompt with classifications
        prompt = self._create_prompt(classifications)
        
        try:
            # Generate story using improved API request method
            translation = self._make_api_request(prompt)
            if not translation:
                translation = "Failed to generate translation due to API issues."
            
            # Evaluate the translation if judge is available
            evaluation = None
            if self.translation_judge and self.translation_judge.rag_initialized:
                try:
                    self.logger.info("Evaluating translation quality...")
                    evaluation = self.translation_judge.evaluate_translation(classifications, translation)
                    self.logger.info(f"Translation evaluation completed. Overall score: {evaluation.overall_score:.1f}/10")
                except Exception as e:
                    self.logger.warning(f"Translation evaluation failed: {e}")
            
            # RAG-based enhancement/refinement (with rate limiting protection)
            enhanced_translation = translation
            enhancement_note = None
            if rag_query is not None and not os.getenv("DISABLE_RAG_REFINEMENT", "").lower() in ("1", "true", "yes"):
                try:
                    symbols_info = self._format_symbols_for_rag(classifications)
                    refine_prompt = (
                        "Refine and improve the following Egyptological translation. Ensure historical and cultural "
                        "accuracy, use proper Egyptological terminology, and clearly weight content by symbol confidence.\n\n"
                        f"Detected Symbols:\n{symbols_info}\n\n"
                        f"Original Translation:\n{translation}\n\n"
                        "Return only the improved translation text."
                    )
                    # Use the same rate limiting logic for RAG queries
                    enhanced_translation = self._make_api_request(refine_prompt) or translation
                    enhancement_note = "rag_refined"
                except Exception as e:
                    self.logger.warning(f"RAG enhancement failed: {e}")
            
            # Return comprehensive results
            result = {
                "translation": enhanced_translation,
                "translation_base": translation,
                "translation_enhancement": enhancement_note,
                "evaluation": evaluation,
                "classifications": classifications
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Story generation failed: {str(e)}")
            return {
                "translation": f"Translation could not be generated due to an error: {str(e)}",
                "evaluation": None,
                "classifications": classifications,
                "error": str(e)
            }

class HieroglyphPipeline:
    """Orchestrates the complete hieroglyph processing workflow"""
    def __init__(self, config: HieroglyphConfig, api_key: str = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # Use OPEN_ROUTER_API_KEY for OpenRouter access; None disables base LLM story
        self.api_key = api_key or os.getenv("OPEN_ROUTER_API_KEY")
        
        # Check for fallback modes
        self.fallback_mode = os.getenv("FALLBACK_MODE", "").lower() in ("1", "true", "yes")
        self.no_llm_mode = os.getenv("NO_LLM_MODE", "").lower() in ("1", "true", "yes")
        
        # Initialize components
        self.image_processor = ImageProcessor(config)
        self.segmentation = SegmentationEngine(config)
        self.classifier = HieroglyphClassifier(config)
        
        # Initialize story generator only if enabled in config and not in no-LLM mode
        self.story_generator = None
        if config.STORY_ENABLED and self.api_key and not self.no_llm_mode:
            self.story_generator = StoryGenerator(config, self.api_key)
        elif self.no_llm_mode:
            self.logger.info("NO_LLM_MODE enabled - story generation disabled")
        elif not self.api_key:
            self.logger.info("No API key provided - story generation disabled")
    
    def setup(self):
        """Initialize all pipeline components"""
        self.logger.info("Initializing pipeline components")
        
        try:
            # Initialize segmentation model (SAM)
            self.logger.debug("Initializing SAM segmentation model")
            self.segmentation.initialize_sam()
            
            # Load classification model and metadata
            self.logger.debug("Loading classifier model")
            self.classifier.load_model()
            self.classifier.load_gardiner_metadata()
            
            # Initialize story generator if enabled
            if self.story_generator:
                self.logger.debug("Initializing story generator")
                self.story_generator.initialize_client()
                self.story_generator.initialize_translation_judge()
                
        except Exception as e:
            self.logger.critical(f"Pipeline initialization failed: {str(e)}")
            raise RuntimeError(f"Pipeline setup error: {e}") from e
    
    def process_image(self, image_path: str, output_dir: str) -> dict:
        """Process a single image through the pipeline"""
        start_time = time.time()
        results = {
            "processing_time": None,
            "image_path": image_path,
            "symbols_found": 0,
            "classifications": [],
            "story": None,
            "output_dir": output_dir,
            "error": None
        }
        
        try:
            # Create output directories
            os.makedirs(output_dir, exist_ok=True)
            symbols_dir = os.path.join(output_dir, "symbols")
            os.makedirs(symbols_dir, exist_ok=True)
            
            # 1. Image processing
            image = self.image_processor.load_and_preprocess(image_path)
            edges = self.image_processor.detect_edges(image)
            
            # 2. Segmentation
            masks = self.segmentation.generate_masks(edges)
            masks = self.segmentation.post_process_masks(masks)
            clusters = self.segmentation.cluster_masks(masks)
            
            # 3. Symbol extraction (tight crop from segmentation mask with padding)
            symbol_paths = []
            height, width = image.shape[:2]
            seen_boxes = set()
            PADDING = 2  # pixels padding around tight mask bbox
            MIN_SIDE = 8  # discard tiny crops
            for i, cluster in enumerate(clusters):
                for j, mask in enumerate(cluster):
                    seg = mask.get('segmentation')
                    if isinstance(seg, np.ndarray) and seg.shape[:2] == (height, width):
                        ys, xs = np.where(seg)
                        if ys.size == 0 or xs.size == 0:
                            continue
                        y0, y1 = max(0, int(ys.min()) - PADDING), min(height, int(ys.max()) + 1 + PADDING)
                        x0, x1 = max(0, int(xs.min()) - PADDING), min(width, int(xs.max()) + 1 + PADDING)
                    else:
                        # fallback to provided bbox
                        bx, by, bw, bh = map(int, mask['bbox'])
                        x0 = max(0, min(bx, width - 1))
                        y0 = max(0, min(by, height - 1))
                        x1 = max(x0 + 1, min(bx + bw, width))
                        y1 = max(y0 + 1, min(by + bh, height))

                    # filter tiny/duplicate boxes
                    if (x1 - x0) < MIN_SIDE or (y1 - y0) < MIN_SIDE:
                        continue
                    box_key = (x0, y0, x1, y1)
                    if box_key in seen_boxes:
                        continue
                    seen_boxes.add(box_key)

                    crop = image[y0:y1, x0:x1].copy()
                    if isinstance(seg, np.ndarray) and seg.shape[:2] == (height, width):
                        seg_crop = seg[y0:y1, x0:x1]
                        bg = np.ones_like(crop) * 255
                        mask3 = np.stack([seg_crop] * 3, axis=-1)
                        crop = np.where(mask3, crop, bg)

                    path = os.path.join(symbols_dir, f"symbol_{i+1}_{j+1}.png")
                    cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    symbol_paths.append(path)
            
            results["symbols_found"] = len(symbol_paths)
            
            # 4. Classification
            if symbol_paths:
                results["classifications"] = self.classifier.classify_symbols(symbol_paths)
            
            # 5. Story generation (if enabled)
            if self.story_generator and results["classifications"]:
                try:
                    story_result = self.story_generator.generate_story(results["classifications"])
                    
                    # Handle new return format with evaluation
                    if isinstance(story_result, dict):
                        results["story"] = story_result.get("translation", "")
                        results["story_base"] = story_result.get("translation_base")
                        results["story_enhancement"] = story_result.get("translation_enhancement")
                        results["evaluation"] = story_result.get("evaluation")
                        results["translation_metadata"] = {
                            "has_evaluation": story_result.get("evaluation") is not None,
                            "error": story_result.get("error")
                        }
                    else:
                        # Fallback for old format
                        results["story"] = story_result
                        results["evaluation"] = None
                        results["translation_metadata"] = {"has_evaluation": False}
                    
                    if not results["story"]:
                        results["error"] = "Story generation produced no output"
                        
                except Exception as e:
                    self.logger.error(f"Story generation failed: {str(e)}")
                    # Use fallback story on error
                    results["story"] = self._generate_fallback_story(results["classifications"])
                    results["story_enhancement"] = "fallback_on_error"
                    results["evaluation"] = None
                    results["translation_metadata"] = {"has_evaluation": False, "error": str(e)}
            elif not self.story_generator:
                # Try RAG-only contextual summary if classifications exist
                results["evaluation"] = None
                results["translation_metadata"] = {"has_evaluation": False}
                if rag_query and results["classifications"] and not os.getenv("DISABLE_RAG_REFINEMENT", "").lower() in ("1", "true", "yes"):
                    try:
                        # Build a creative narrative even without base LLM
                        symbols_text = StoryGenerator(self.config, api_key="dummy")._format_symbols_for_tour_guide(results["classifications"])  # aggregated counts only
                        ctx_prompt = (
                            "Craft a creative, engaging tour‑guide style story for visitors based on the detection "
                            "counts below. Do NOT list symbols, codes, or confidence numbers in the story; write a "
                            "cohesive narrative in 3–6 short paragraphs, accessible to non‑experts, highlighting likely "
                            "meaning, function, historical context, and why it matters.\n\n"
                            f"Reference (counts only — do not list these):\n{symbols_text}"
                        )
                        # Use rate limiting for RAG queries too
                        story_gen = StoryGenerator(self.config, api_key="dummy")
                        story_gen.client = True  # Enable for RAG-only mode
                        results["story"] = story_gen._make_api_request(ctx_prompt)
                        results["story_enhancement"] = "rag_context_only"
                    except Exception as e:
                        self.logger.warning(f"RAG contextual summary failed: {e}")
                        # Use fallback story if RAG fails
                        results["story"] = self._generate_fallback_story(results["classifications"])
                        results["story_enhancement"] = "fallback_rag_failed"
                else:
                    # No LLM available, use fallback story
                    results["story"] = self._generate_fallback_story(results["classifications"])
                    results["story_enhancement"] = "fallback_no_llm"
            
            # Final processing time
            results["processing_time"] = time.time() - start_time
            
            # Save results
            # Ensure evaluation is JSON-serializable when writing pipeline-local results
            if results.get("evaluation") is not None and not isinstance(results["evaluation"], dict):
                eval_obj = results["evaluation"]
                try:
                    results["evaluation"] = {
                        "overall_score": eval_obj.overall_score,
                        "criteria_scores": {k.value: v for k, v in eval_obj.criteria_scores.items()},
                        "strengths": eval_obj.strengths,
                        "weaknesses": eval_obj.weaknesses,
                        "suggestions": eval_obj.suggestions,
                        "historical_context": eval_obj.historical_context,
                        "confidence_analysis": eval_obj.confidence_analysis,
                    }
                except Exception:
                    # Best effort: drop non-serializable evaluation if conversion fails
                    results["evaluation"] = None
            with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            return results
        
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
            return results
    
    def _generate_fallback_story(self, classifications: list) -> str:
        """Generate a basic fallback story when LLM is unavailable"""
        if not classifications:
            return "No symbols detected for translation."
        
        # Count symbols by confidence
        high_conf = [c for c in classifications if c.get('confidence', 0) > 0.7]
        medium_conf = [c for c in classifications if 0.4 <= c.get('confidence', 0) <= 0.7]
        low_conf = [c for c in classifications if c.get('confidence', 0) < 0.4]
        
        story_parts = []
        
        if high_conf:
            symbols = [c.get('Gardiner Code', 'Unknown') for c in high_conf]
            story_parts.append(f"High confidence symbols detected: {', '.join(symbols)}")
        
        if medium_conf:
            symbols = [c.get('Gardiner Code', 'Unknown') for c in medium_conf]
            story_parts.append(f"Medium confidence symbols: {', '.join(symbols)}")
        
        if low_conf:
            symbols = [c.get('Gardiner Code', 'Unknown') for c in low_conf]
            story_parts.append(f"Low confidence symbols: {', '.join(symbols)}")
        
        # Add basic context
        total_symbols = len(classifications)
        story_parts.append(f"Total symbols detected: {total_symbols}")
        story_parts.append("Note: Full translation requires LLM access. Enable API key for complete analysis.")
        
        return " | ".join(story_parts)