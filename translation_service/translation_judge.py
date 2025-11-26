#!/usr/bin/env python3
"""
RAG-based Translation Judge System
Evaluates hieroglyph translation quality using historical context and Egyptology knowledge
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
RAG_SRC = REPO_ROOT / "Agentic_RAG" / "src"
if RAG_SRC.exists():
    sys.path.append(str(RAG_SRC))

from pipeline.model import get_document_stats, load_documents_from_data_dir, rag_query

logger = logging.getLogger(__name__)

class EvaluationCriteria(Enum):
    """Evaluation criteria for translation quality"""
    HISTORICAL_ACCURACY = "historical_accuracy"
    CULTURAL_CONTEXT = "cultural_context"
    SYMBOL_MEANING_ALIGNMENT = "symbol_meaning_alignment"
    NARRATIVE_COHERENCE = "narrative_coherence"
    EGYPTOLOGICAL_TERMINOLOGY = "egyptological_terminology"
    CONFIDENCE_WEIGHTING = "confidence_weighting"

@dataclass
class TranslationEvaluation:
    """Results of translation evaluation"""
    overall_score: float
    criteria_scores: Dict[EvaluationCriteria, float]
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    historical_context: str
    confidence_analysis: str

class RAGTranslationJudge:
    """RAG-based system for evaluating hieroglyph translations"""
    
    def __init__(self):
        self.rag_initialized = False
        self._initialize_rag_system()
        
        # Evaluation prompts for different criteria
        self.evaluation_prompts = {
            EvaluationCriteria.HISTORICAL_ACCURACY: """
            As an expert Egyptologist, evaluate the historical accuracy of this hieroglyph translation:
            
            Detected Symbols: {symbols_info}
            Generated Translation: {translation}
            
            CRITICAL: You MUST provide a numerical score from 0-10 and detailed feedback.
            
            Rate the historical accuracy (0-10) and provide specific feedback on:
            1. Historical period accuracy - Are references appropriate for the time period?
            2. Pharaoh/dynasty references - Are names, titles, and dynasties correct?
            3. Cultural practices mentioned - Do practices match the historical period?
            4. Timeline consistency - Are events and references chronologically accurate?
            5. Anachronisms - Are there any modern concepts inappropriately placed?
            
            FORMAT YOUR RESPONSE EXACTLY AS:
            Score: [number from 0-10]
            Feedback: [detailed analysis of historical accuracy, specific examples of what is correct or incorrect]
            """,
            
            EvaluationCriteria.CULTURAL_CONTEXT: """
            Evaluate the cultural context and authenticity of this hieroglyph translation:
            
            Detected Symbols: {symbols_info}
            Generated Translation: {translation}
            
            CRITICAL: You MUST provide a numerical score from 0-10 and detailed feedback.
            
            Assess (0-10) and comment on:
            1. Ancient Egyptian cultural practices - Are practices authentic and period-appropriate?
            2. Religious beliefs and rituals - Do religious references match ancient Egyptian beliefs?
            3. Social hierarchy references - Are social structures accurately represented?
            4. Daily life accuracy - Do descriptions match ancient Egyptian daily life?
            5. Symbolic meanings - Are cultural symbols interpreted correctly?
            
            FORMAT YOUR RESPONSE EXACTLY AS:
            Score: [number from 0-10]
            Feedback: [detailed analysis of cultural authenticity, specific examples of what is culturally appropriate or inappropriate]
            """,
            
            EvaluationCriteria.SYMBOL_MEANING_ALIGNMENT: """
            Analyze how well the translation aligns with the detected hieroglyph symbols:
            
            Detected Symbols: {symbols_info}
            Generated Translation: {translation}
            
            CRITICAL: You MUST provide a numerical score from 0-10 and detailed feedback.
            
            Evaluate (0-10) the alignment between:
            1. Symbol meanings and translation content - Do the symbols' meanings appear in the translation?
            2. Confidence levels and emphasis in translation - Are high-confidence symbols emphasized more?
            3. Symbol combinations and narrative flow - Do symbol combinations create coherent meaning?
            4. Missing or overemphasized symbols - Are important symbols included, unimportant ones de-emphasized?
            5. Symbol interpretation accuracy - Are the symbols interpreted correctly according to their meanings?
            
            FORMAT YOUR RESPONSE EXACTLY AS:
            Score: [number from 0-10]
            Feedback: [detailed analysis of symbol-translation alignment, specific examples of good or poor alignment]
            """,
            
            EvaluationCriteria.NARRATIVE_COHERENCE: """
            Assess the narrative coherence and storytelling quality:
            
            Detected Symbols: {symbols_info}
            Generated Translation: {translation}
            
            CRITICAL: You MUST provide a numerical score from 0-10 and detailed feedback.
            
            Rate (0-10) the translation's:
            1. Story structure and flow - Does the narrative have a clear beginning, middle, and end?
            2. Logical progression of ideas - Do ideas flow logically from one to the next?
            3. Ancient Egyptian narrative style - Does it match ancient Egyptian storytelling conventions?
            4. Engagement and readability - Is it engaging and easy to follow?
            5. Coherence and consistency - Is the story internally consistent and coherent?
            
            FORMAT YOUR RESPONSE EXACTLY AS:
            Score: [number from 0-10]
            Feedback: [detailed analysis of narrative quality, specific examples of good or poor storytelling elements]
            """,
            
            EvaluationCriteria.EGYPTOLOGICAL_TERMINOLOGY: """
            Evaluate the use of proper Egyptological terminology:
            
            Detected Symbols: {symbols_info}
            Generated Translation: {translation}
            
            CRITICAL: You MUST provide a numerical score from 0-10 and detailed feedback.
            
            Assess (0-10) the accuracy of:
            1. Egyptian names and titles - Are pharaoh names, titles, and god names correct?
            2. Technical terms and concepts - Are Egyptological terms used correctly?
            3. Hieroglyphic sign names - Are hieroglyph names and Gardiner codes accurate?
            4. Historical period terminology - Is period-specific language used appropriately?
            5. Scholarly precision - Is the language scholarly and precise?
            
            FORMAT YOUR RESPONSE EXACTLY AS:
            Score: [number from 0-10]
            Feedback: [detailed analysis of terminology accuracy, specific examples of correct or incorrect usage]
            """,
            
            EvaluationCriteria.CONFIDENCE_WEIGHTING: """
            Analyze how well the translation weights symbol confidence levels:
            
            Detected Symbols: {symbols_info}
            Generated Translation: {translation}
            
            CRITICAL: You MUST provide a numerical score from 0-10 and detailed feedback.
            
            Evaluate (0-10) whether:
            1. High-confidence symbols are properly emphasized - Are symbols with >70% confidence featured prominently?
            2. Low-confidence symbols are appropriately de-emphasized - Are symbols with <40% confidence downplayed?
            3. Confidence levels inform translation decisions - Does the translation reflect confidence levels?
            4. Uncertainty is properly communicated - Is uncertainty about low-confidence symbols acknowledged?
            5. Appropriate weighting - Is the balance between high/medium/low confidence symbols appropriate?
            
            FORMAT YOUR RESPONSE EXACTLY AS:
            Score: [number from 0-10]
            Feedback: [detailed analysis of confidence weighting, specific examples of appropriate or inappropriate weighting]
            """
        }
    
    def _initialize_rag_system(self):
        """Initialize the RAG system for evaluation"""
        try:
            logger.info("Initializing RAG system for translation evaluation...")
            load_result = load_documents_from_data_dir()
            
            if load_result["success"]:
                self.rag_initialized = True
                logger.info(f"RAG system initialized with {load_result['files_processed']} files")
            else:
                logger.error(f"Failed to initialize RAG system: {load_result.get('error', 'Unknown error')}")
                self.rag_initialized = False
                
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            self.rag_initialized = False
    
    def _format_symbols_info(self, classifications: List[Dict]) -> str:
        """Format symbol information for evaluation prompts"""
        if not classifications:
            return "No symbols detected"
        
        formatted_symbols = []
        for cls in classifications:
            if 'error' not in cls:
                confidence = cls.get('confidence', 0)
                symbol_info = (
                    f"• {cls.get('Hieroglyph', '?')} "
                    f"(Gardiner: {cls.get('Gardiner Code', 'Unknown')}, "
                    f"Confidence: {confidence:.1%})\n"
                    f"  Meaning: {cls.get('Description', 'Unknown')}\n"
                    f"  Details: {cls.get('Details', 'No additional details')}"
                )
                formatted_symbols.append(symbol_info)
        
        return "\n\n".join(formatted_symbols)
    
    def _evaluate_criterion(self, criterion: EvaluationCriteria, symbols_info: str, translation: str) -> Tuple[float, str]:
        """Evaluate a specific criterion using RAG system"""
        if not self.rag_initialized:
            return 0.0, "RAG system not initialized"
        
        try:
            prompt = self.evaluation_prompts[criterion].format(
                symbols_info=symbols_info,
                translation=translation
            )
            
            # Use RAG system to get evaluation
            response = rag_query(prompt)
            
            # Parse score and feedback from response
            score = self._extract_score(response)
            feedback = self._extract_feedback(response)
            
            return score, feedback
            
        except Exception as e:
            logger.error(f"Error evaluating {criterion.value}: {e}")
            return 0.0, f"Evaluation error: {str(e)}"
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from RAG response with improved patterns"""
        import re
        
        # Enhanced score patterns to catch more variations
        score_patterns = [
            r'Score:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)/10',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*10',
            r'Rating:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*points?',
            r'(\d+(?:\.\d+)?)\s*\/\s*10',
            r'(\d+(?:\.\d+)?)\s*of\s*10',
            r'(\d+(?:\.\d+)?)\s*\/10',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*ten',
            r'(\d+(?:\.\d+)?)\s*\/\s*ten'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 10.0)  # Clamp between 0-10
        
        # Look for percentage scores and convert to 0-10 scale
        percentage_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*percent'
        ]
        
        for pattern in percentage_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                percentage = float(match.group(1))
                score = (percentage / 100.0) * 10.0  # Convert percentage to 0-10 scale
                return min(max(score, 0.0), 10.0)
        
        # Look for qualitative assessments and convert to scores
        qualitative_scores = {
            'excellent': 9.0, 'outstanding': 9.5, 'exceptional': 9.5,
            'very good': 8.0, 'very high': 8.0, 'strong': 8.0,
            'good': 7.0, 'solid': 7.0, 'adequate': 6.0,
            'fair': 5.0, 'average': 5.0, 'moderate': 5.0,
            'poor': 3.0, 'weak': 3.0, 'inadequate': 2.0,
            'very poor': 1.0, 'terrible': 1.0, 'failing': 1.0
        }
        
        response_lower = response.lower()
        for qual, score in qualitative_scores.items():
            if qual in response_lower:
                return score
        
        # If no score found, try to extract any number (but be more selective)
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if numbers:
            # Take the first reasonable number (likely to be a score)
            for num_str in numbers:
                score = float(num_str)
                if 0 <= score <= 10:  # Only accept numbers in valid score range
                    return score
        
        # If still no score found, analyze content quality for intelligent default
        return self._analyze_content_quality(response)
    
    def _analyze_content_quality(self, response: str) -> float:
        """Analyze response content to determine quality score when no explicit score is found"""
        response_lower = response.lower()
        
        # Positive indicators
        positive_indicators = [
            'excellent', 'outstanding', 'very good', 'strong', 'accurate', 'precise',
            'well-written', 'comprehensive', 'detailed', 'thorough', 'impressive',
            'high quality', 'professional', 'scholarly', 'authentic', 'historically accurate'
        ]
        
        # Negative indicators
        negative_indicators = [
            'poor', 'weak', 'inadequate', 'inaccurate', 'incorrect', 'wrong',
            'lacking', 'insufficient', 'vague', 'unclear', 'confusing', 'misleading',
            'anachronistic', 'modern', 'inappropriate', 'unprofessional'
        ]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        # Calculate base score with more realistic distribution
        if positive_count > negative_count:
            base_score = 5.5 + min(positive_count * 0.3, 2.5)  # 5.5-8.0 range
        elif negative_count > positive_count:
            base_score = 4.0 - min(negative_count * 0.3, 2.0)  # 2.0-4.0 range
        else:
            base_score = 4.5  # Slightly below neutral for realism
        
        # Adjust based on response length and detail
        word_count = len(response.split())
        if word_count > 100:  # Detailed response
            base_score += 0.3
        elif word_count < 50:  # Brief response
            base_score -= 0.3
        
        # Add some randomness to avoid perfect scores
        import random
        random_factor = random.uniform(-0.5, 0.5)
        base_score += random_factor
        
        return min(max(base_score, 1.0), 8.5)  # Clamp between 1.0-8.5 (no perfect 10s)
    
    def _extract_feedback(self, response: str) -> str:
        """Extract feedback text from RAG response"""
        # Look for feedback sections
        feedback_patterns = [
            r'Feedback:\s*(.+?)(?:\n\n|\Z)',
            r'Analysis:\s*(.+?)(?:\n\n|\Z)',
            r'Comments:\s*(.+?)(?:\n\n|\Z)'
        ]
        
        for pattern in feedback_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no specific feedback section, return the whole response
        return response.strip()
    
    def evaluate_translation(self, classifications: List[Dict], translation: str) -> TranslationEvaluation:
        """Comprehensive evaluation of a hieroglyph translation"""
        if not self.rag_initialized:
            logger.warning("RAG system not initialized, using fallback evaluation")
            return self._fallback_evaluation(classifications, translation)
        
        logger.info("Starting comprehensive translation evaluation...")
        
        # Format symbols information
        symbols_info = self._format_symbols_info(classifications)
        
        # Evaluate each criterion
        criteria_scores = {}
        all_feedback = []
        
        for criterion in EvaluationCriteria:
            logger.info(f"Evaluating {criterion.value}...")
            score, feedback = self._evaluate_criterion(criterion, symbols_info, translation)
            criteria_scores[criterion] = score
            all_feedback.append(f"{criterion.value}: {feedback}")
        
        # Calculate overall score (weighted average)
        weights = {
            EvaluationCriteria.HISTORICAL_ACCURACY: 0.25,
            EvaluationCriteria.CULTURAL_CONTEXT: 0.20,
            EvaluationCriteria.SYMBOL_MEANING_ALIGNMENT: 0.20,
            EvaluationCriteria.NARRATIVE_COHERENCE: 0.15,
            EvaluationCriteria.EGYPTOLOGICAL_TERMINOLOGY: 0.10,
            EvaluationCriteria.CONFIDENCE_WEIGHTING: 0.10
        }
        
        overall_score = sum(score * weights[criterion] for criterion, score in criteria_scores.items())
        
        # Generate strengths, weaknesses, and suggestions
        strengths, weaknesses, suggestions = self._analyze_scores(criteria_scores, all_feedback)
        
        # Get historical context
        historical_context = self._get_historical_context(classifications)
        
        # Analyze confidence weighting
        confidence_analysis = self._analyze_confidence_weighting(classifications, translation)
        
        return TranslationEvaluation(
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            historical_context=historical_context,
            confidence_analysis=confidence_analysis
        )
    
    def _analyze_scores(self, criteria_scores: Dict[EvaluationCriteria, float], feedback: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Analyze scores to generate strengths, weaknesses, and suggestions"""
        strengths = []
        weaknesses = []
        suggestions = []
        
        for criterion, score in criteria_scores.items():
            if score >= 8.0:
                strengths.append(f"Excellent {criterion.value.replace('_', ' ')} (Score: {score:.1f})")
            elif score >= 6.0:
                strengths.append(f"Good {criterion.value.replace('_', ' ')} (Score: {score:.1f})")
            elif score >= 4.0:
                weaknesses.append(f"Needs improvement in {criterion.value.replace('_', ' ')} (Score: {score:.1f})")
            else:
                weaknesses.append(f"Poor {criterion.value.replace('_', ' ')} (Score: {score:.1f})")
        
        # Generate suggestions based on lowest scores
        sorted_criteria = sorted(criteria_scores.items(), key=lambda x: x[1])
        for criterion, score in sorted_criteria[:3]:  # Top 3 areas for improvement
            if score < 7.0:
                suggestions.append(f"Focus on improving {criterion.value.replace('_', ' ')} - current score: {score:.1f}")
        
        return strengths, weaknesses, suggestions
    
    def _get_historical_context(self, classifications: List[Dict]) -> str:
        """Get historical context for the detected symbols"""
        if not self.rag_initialized:
            return "Historical context unavailable - RAG system not initialized"
        
        try:
            # Extract key symbols for context query
            symbols = [cls.get('Hieroglyph', '') for cls in classifications if 'error' not in cls]
            if not symbols:
                return "No symbols detected for historical context"
            
            context_query = f"What is the historical significance and context of these Egyptian hieroglyphs: {', '.join(symbols[:5])}?"
            context = rag_query(context_query)
            return context[:500] + "..." if len(context) > 500 else context
            
        except Exception as e:
            logger.error(f"Error getting historical context: {e}")
            return f"Error retrieving historical context: {str(e)}"
    
    def _analyze_confidence_weighting(self, classifications: List[Dict], translation: str) -> str:
        """Analyze how well confidence levels are weighted in the translation"""
        if not classifications:
            return "No symbols to analyze confidence weighting"
        
        high_conf = [cls for cls in classifications if cls.get('confidence', 0) > 0.7]
        medium_conf = [cls for cls in classifications if 0.4 <= cls.get('confidence', 0) <= 0.7]
        low_conf = [cls for cls in classifications if cls.get('confidence', 0) < 0.4]
        
        analysis = f"Confidence Distribution: {len(high_conf)} high, {len(medium_conf)} medium, {len(low_conf)} low confidence symbols\n"
        
        if high_conf:
            analysis += f"High confidence symbols should be emphasized: {[cls.get('Hieroglyph', '?') for cls in high_conf]}\n"
        
        if low_conf:
            analysis += f"Low confidence symbols should be de-emphasized: {[cls.get('Hieroglyph', '?') for cls in low_conf]}\n"
        
        return analysis
    
    def _fallback_evaluation(self, classifications: List[Dict], translation: str) -> TranslationEvaluation:
        """Fallback evaluation when RAG system is not available"""
        logger.info("Performing fallback evaluation without RAG system")
        
        # Basic evaluation based on translation characteristics
        criteria_scores = {}
        
        # Historical Accuracy - basic check for ancient Egyptian terms
        historical_terms = ['pharaoh', 'dynasty', 'ancient', 'egyptian', 'temple', 'pyramid', 'gods', 'afterlife']
        historical_score = min(10.0, len([term for term in historical_terms if term.lower() in translation.lower()]) * 1.5)
        criteria_scores[EvaluationCriteria.HISTORICAL_ACCURACY] = historical_score
        
        # Cultural Context - check for cultural terms
        cultural_terms = ['ma\'at', 'ka', 'ba', 'ankh', 'eye of horus', 'scarab', 'mummy', 'tomb']
        cultural_score = min(10.0, len([term for term in cultural_terms if term.lower() in translation.lower()]) * 1.5)
        criteria_scores[EvaluationCriteria.CULTURAL_CONTEXT] = cultural_score
        
        # Symbol Meaning Alignment - check if translation mentions symbols
        symbol_mentions = len([cls for cls in classifications if 'error' not in cls and cls.get('Hieroglyph', '') in translation])
        total_symbols = len([cls for cls in classifications if 'error' not in cls])
        alignment_score = (symbol_mentions / max(total_symbols, 1)) * 10.0
        criteria_scores[EvaluationCriteria.SYMBOL_MEANING_ALIGNMENT] = alignment_score
        
        # Narrative Coherence - basic length and structure check
        word_count = len(translation.split())
        coherence_score = min(10.0, max(0.0, (word_count - 50) / 20))  # Scale based on length
        criteria_scores[EvaluationCriteria.NARRATIVE_COHERENCE] = coherence_score
        
        # Egyptological Terminology - check for proper terms
        egyptological_terms = ['hieroglyph', 'cartouche', 'papyrus', 'sarcophagus', 'obelisk']
        terminology_score = min(10.0, len([term for term in egyptological_terms if term.lower() in translation.lower()]) * 2.0)
        criteria_scores[EvaluationCriteria.EGYPTOLOGICAL_TERMINOLOGY] = terminology_score
        
        # Confidence Weighting - check if high confidence symbols are emphasized
        high_conf_symbols = [cls for cls in classifications if cls.get('confidence', 0) > 0.7]
        if high_conf_symbols:
            high_conf_mentioned = len([cls for cls in high_conf_symbols if cls.get('Hieroglyph', '') in translation])
            confidence_score = (high_conf_mentioned / len(high_conf_symbols)) * 10.0
        else:
            confidence_score = 5.0  # Neutral if no high confidence symbols
        criteria_scores[EvaluationCriteria.CONFIDENCE_WEIGHTING] = confidence_score
        
        # Calculate overall score
        weights = {
            EvaluationCriteria.HISTORICAL_ACCURACY: 0.25,
            EvaluationCriteria.CULTURAL_CONTEXT: 0.20,
            EvaluationCriteria.SYMBOL_MEANING_ALIGNMENT: 0.20,
            EvaluationCriteria.NARRATIVE_COHERENCE: 0.15,
            EvaluationCriteria.EGYPTOLOGICAL_TERMINOLOGY: 0.10,
            EvaluationCriteria.CONFIDENCE_WEIGHTING: 0.10
        }
        
        overall_score = sum(score * weights[criterion] for criterion, score in criteria_scores.items())
        
        # Generate basic feedback
        strengths = []
        weaknesses = []
        suggestions = []
        
        if overall_score >= 7.0:
            strengths.append("Good use of ancient Egyptian terminology and cultural references")
        elif overall_score >= 5.0:
            strengths.append("Some appropriate historical and cultural elements present")
        else:
            weaknesses.append("Limited use of appropriate ancient Egyptian terminology and cultural context")
        
        if alignment_score < 5.0:
            weaknesses.append("Translation may not align well with detected symbols")
            suggestions.append("Focus on incorporating meanings of detected symbols into the narrative")
        
        if historical_score < 5.0:
            weaknesses.append("Limited historical accuracy and period-appropriate references")
            suggestions.append("Include more specific historical periods, dynasties, or pharaohs")
        
        if not suggestions:
            suggestions.append("Consider enhancing with more specific Egyptological terminology and cultural context")
        
        return TranslationEvaluation(
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            historical_context="Fallback evaluation - RAG system not available for detailed historical context",
            confidence_analysis=f"Detected {len(high_conf_symbols)} high-confidence symbols out of {len(classifications)} total symbols"
        )

# Example usage and testing
if __name__ == "__main__":
    # Test the translation judge
    judge = RAGTranslationJudge()
    
    # Sample classifications (mock data)
    sample_classifications = [
        {
            'Hieroglyph': '𓂀',
            'Gardiner Code': 'A1',
            'Description': 'Egyptian vulture',
            'Details': 'Represents the sound "a" and the concept of "mother"',
            'confidence': 0.85
        },
        {
            'Hieroglyph': '𓃭',
            'Gardiner Code': 'E1',
            'Description': 'Lion',
            'Details': 'Symbol of strength and royalty',
            'confidence': 0.72
        }
    ]
    
    sample_translation = "The mother goddess, represented by the vulture, watches over the kingdom with the strength of a lion, symbolizing divine protection and royal power."
    
    if judge.rag_initialized:
        evaluation = judge.evaluate_translation(sample_classifications, sample_translation)
        
        print("Translation Evaluation Results:")
        print(f"Overall Score: {evaluation.overall_score:.1f}/10")
        print("\nCriteria Scores:")
        for criterion, score in evaluation.criteria_scores.items():
            print(f"  {criterion.value}: {score:.1f}/10")
        
        print(f"\nStrengths: {evaluation.strengths}")
        print(f"Weaknesses: {evaluation.weaknesses}")
        print(f"Suggestions: {evaluation.suggestions}")
    else:
        print("RAG system not initialized. Cannot perform evaluation.")

