"""
Query Rewriter Service - Two-stage query rewriting for RAG optimization.

Stage 1: Rewrite user queries for optimal vector database retrieval
Stage 2: Rewrite prompts for natural, conversational LLM responses
"""

import re
from typing import List, Dict, Optional
import os
from langchain_core.prompts import PromptTemplate


class QueryRewriterService:
    """
    Two-stage query rewriting service for RAG optimization.
    
    Stage 1: Converts natural language queries into retrieval-optimized queries
    Stage 2: Converts retrieved context into conversational tour guide responses
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the Query Rewriter Service.
        
        Args:
            llm_client: Optional LLM client for advanced rewriting. If None, uses rule-based approach.
        """
        self.llm_client = llm_client
        
        # Common filler words to remove in retrieval queries
        self.filler_words = {
            'please', 'could', 'would', 'can', 'you', 'tell', 'me', 'about',
            'what', 'is', 'are', 'the', 'a', 'an', 'i', 'want', 'to', 'know',
            'explain', 'describe', 'show', 'give', 'hi', 'hello', 'hey',
            'thanks', 'thank', 'you'
        }
        
        # Important keywords to preserve
        self.important_keywords = {
            'pharaoh', 'pyramid', 'temple', 'tomb', 'mummy', 'hieroglyph',
            'sphinx', 'nile', 'delta', 'dynasty', 'god', 'goddess', 'king',
            'queen', 'papyrus', 'scarab', 'ankh', 'obelisk', 'sarcophagus',
            'ramses', 'tutankhamun', 'cleopatra', 'khufu', 'osiris', 'isis',
            'ra', 'anubis', 'horus', 'karnak', 'luxor', 'giza', 'valley',
            'kings', 'alexandria', 'cairo', 'aswan', 'abu', 'simbel'
        }
    
    def rewrite_for_retrieval(self, user_query: str) -> str:
        """
        Stage 1: Rewrite user query for optimal vector database retrieval.
        
        Converts natural conversational queries into compact, keyword-focused queries
        that are optimized for semantic similarity search.
        
        Example:
            Input:  "Hi! Can you please tell me about the famous pharaohs of ancient Egypt?"
            Output: "famous pharaohs ancient Egypt"
            
            Input:  "I'm curious about how the pyramids were built back in the day"
            Output: "pyramids construction building methods"
        
        Args:
            user_query: Raw natural language query from user
            
        Returns:
            Optimized query string for vector database search
        """
        if not user_query or not user_query.strip():
            return ""
        
        # Store original for fallback
        original_query = user_query.strip()
        
        # Convert to lowercase for processing
        query = original_query.lower()
        
        # Remove punctuation except hyphens (for names like Abu-Simbel)
        query = re.sub(r'[^\w\s\-]', ' ', query)
        
        # Tokenize
        words = query.split()
        
        # Remove filler words but preserve important keywords
        filtered_words = []
        for word in words:
            word_clean = word.strip('-')
            # Keep if it's an important keyword or not a filler word
            if (word_clean in self.important_keywords or 
                word_clean not in self.filler_words or
                len(word_clean) > 6):  # Keep longer words even if not in important list
                filtered_words.append(word_clean)
        
        # Handle special patterns
        rewritten_query = ' '.join(filtered_words)
        
        # Expand common abbreviations and synonyms for better retrieval
        rewritten_query = self._expand_synonyms(rewritten_query)
        
        # Remove extra spaces
        rewritten_query = ' '.join(rewritten_query.split())
        
        # If we filtered too much, fall back to original
        if len(rewritten_query) < 5 and len(original_query) > 10:
            # Keep only essential words from original
            words = original_query.lower().split()
            filtered = [w for w in words if len(w) > 3]
            rewritten_query = ' '.join(filtered[:7])  # Max 7 words
        
        return rewritten_query.strip()
    
    def _expand_synonyms(self, query: str) -> str:
        """
        Expand common synonyms and related terms for better semantic matching.
        
        Args:
            query: Query string to expand
            
        Returns:
            Expanded query with synonyms
        """
        # Synonym mapping for Ancient Egypt domain
        synonym_map = {
            'built': 'construction building',
            'build': 'construction building',
            'made': 'created construction',
            'king': 'pharaoh ruler',
            'ruler': 'pharaoh king',
            'writing': 'hieroglyphs script',
            'gods': 'deities mythology',
            'death': 'afterlife burial',
            'burial': 'tomb mummy afterlife',
            'river': 'nile water',
        }
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            # Add synonyms if available
            if word in synonym_map:
                expanded_words.append(synonym_map[word])
        
        return ' '.join(expanded_words)
    
    def rewrite_for_response(
        self, 
        user_query: str, 
        retrieved_context: List[str],
        language: str = "en"
    ) -> str:
        """
        Stage 2: Create a conversational prompt for the LLM using retrieved context.
        
        Transforms the user query and retrieved context into a friendly, engaging
        prompt that guides the LLM to respond as an expert tour guide.
        
        Example:
            Input:
                user_query: "Tell me about the pyramids"
                retrieved_context: ["The Great Pyramid was built...", "Pyramids served as tombs..."]
                language: "en"
            Output:
                "You are an expert Ancient Egypt tour guide. A visitor asks: 'Tell me about
                the pyramids.' Using the following historical information, provide a warm,
                engaging response..."
        
        Args:
            user_query: Original user query
            retrieved_context: List of relevant text chunks from vector database
            language: Response language ('en' for English, 'ar' for Arabic)
            
        Returns:
            Complete prompt for LLM to generate tour guide response
        """
        # Determine language-specific instructions
        if language.lower() in ['ar', 'arabic', 'عربي']:
            lang_instruction = "Respond in Arabic (العربية) with clear, friendly language."
            greeting_style = "Use warm Arabic greetings and phrases."
        else:
            lang_instruction = "Respond in English with clear, friendly language."
            greeting_style = "Use warm, welcoming tour guide language."
        
        # Format retrieved context
        if retrieved_context:
            context_text = "\n\n".join([
                f"Source {i+1}: {ctx[:500]}..." if len(ctx) > 500 else f"Source {i+1}: {ctx}"
                for i, ctx in enumerate(retrieved_context[:5])  # Max 5 sources
            ])
        else:
            context_text = "No specific historical sources available."
        
        # Detect query intent for tailored instructions
        query_lower = user_query.lower()
        intent_instruction = self._detect_intent(query_lower)
        
        # Build the complete prompt
        prompt = f"""You are a knowledgeable and enthusiastic Ancient Egypt tour guide at a world-class museum. You love sharing fascinating stories about pharaohs, pyramids, hieroglyphs, and ancient Egyptian culture.

**Visitor's Question:**
"{user_query}"

**Historical Context (from museum archives):**
{context_text}

**Your Task:**
{intent_instruction}

**Response Guidelines:**
1. {greeting_style}
2. Be informative yet conversational - like talking to a curious friend
3. Use the retrieved historical context to provide accurate information
4. Add interesting fun facts or stories when relevant
5. Keep responses clear and well-organized (use bullet points if listing multiple items)
6. If the question asks about multiple things, address each one
7. {lang_instruction}
8. If the context doesn't fully answer the question, say what you know and suggest related topics
9. Aim for 3-5 sentences for simple questions, longer for complex topics

**Tone:**
Warm, engaging, educational, and passionate about Ancient Egypt!

Now provide your response:"""

        return prompt
    
    def _detect_intent(self, query: str) -> str:
        """
        Detect the user's query intent to tailor the response instruction.
        
        Args:
            query: Lowercase user query
            
        Returns:
            Specific instruction based on detected intent
        """
        # Define intent patterns
        if any(word in query for word in ['who', 'pharaoh', 'ruler', 'king', 'queen']):
            return "Focus on the person(s) - their life, achievements, and historical significance."
        
        elif any(word in query for word in ['how', 'built', 'build', 'made', 'create', 'construct']):
            return "Explain the process, methods, and techniques used. Include fascinating details about ancient engineering."
        
        elif any(word in query for word in ['why', 'reason', 'purpose', 'significance']):
            return "Explain the reasons, purposes, and cultural/religious significance."
        
        elif any(word in query for word in ['when', 'date', 'period', 'time', 'dynasty']):
            return "Provide historical timeline and context. Mention the dynasty or time period."
        
        elif any(word in query for word in ['where', 'location', 'place', 'site']):
            return "Describe the location, geographical context, and what visitors can see there today."
        
        elif any(word in query for word in ['compare', 'difference', 'similar', 'versus', 'vs']):
            return "Compare and contrast the items clearly. Highlight key similarities and differences."
        
        elif any(word in query for word in ['gods', 'goddess', 'mythology', 'religion', 'belief']):
            return "Explain the religious/mythological aspects with engaging stories from ancient beliefs."
        
        else:
            return "Provide a comprehensive, engaging answer covering the key aspects of the topic."
    
    def rewrite_for_multilingual_retrieval(
        self, 
        user_query: str, 
        target_languages: List[str] = ['en', 'ar']
    ) -> Dict[str, str]:
        """
        Advanced: Rewrite query for multilingual retrieval.
        
        Useful if your vector database contains documents in multiple languages.
        
        Args:
            user_query: Original query
            target_languages: List of language codes to generate queries for
            
        Returns:
            Dictionary mapping language codes to rewritten queries
        """
        # Base rewrite
        base_query = self.rewrite_for_retrieval(user_query)
        
        queries = {
            'en': base_query,
        }
        
        # Add Arabic transliteration if needed
        if 'ar' in target_languages:
            # Simple transliteration mapping (can be enhanced with proper translation API)
            ar_keywords = self._add_arabic_keywords(base_query)
            queries['ar'] = f"{base_query} {ar_keywords}".strip()
        
        return queries
    
    def _add_arabic_keywords(self, query: str) -> str:
        """
        Add Arabic transliterations of common Ancient Egypt terms.
        
        Args:
            query: English query
            
        Returns:
            Arabic keywords to append
        """
        # Mapping of common terms (could be expanded or use translation API)
        arabic_terms = {
            'pyramid': 'هرم',
            'pharaoh': 'فرعون',
            'sphinx': 'أبو الهول',
            'nile': 'النيل',
            'mummy': 'مومياء',
            'temple': 'معبد',
            'hieroglyph': 'هيروغليفية',
        }
        
        query_lower = query.lower()
        arabic_additions = []
        
        for eng_term, ar_term in arabic_terms.items():
            if eng_term in query_lower:
                arabic_additions.append(ar_term)
        
        return ' '.join(arabic_additions)


# Example usage and testing
if __name__ == "__main__":
    # Initialize service
    rewriter = QueryRewriterService()
    
    print("="*60)
    print("STAGE 1: QUERY REWRITING FOR RETRIEVAL")
    print("="*60)
    
    # Test cases for Stage 1
    test_queries = [
        "Hi! Can you please tell me about the famous pharaohs of ancient Egypt?",
        "I'm curious about how the pyramids were built back in the day",
        "What were the religious beliefs of ancient Egyptians?",
        "Tell me about Tutankhamun's tomb discovery",
        "Why did they mummify people?",
        "Compare the pyramids of Giza with other pyramids",
        "My left shoulder hurts and I have numbness in my thumb and index finger"  # Medical analogy example
    ]
    
    for query in test_queries:
        rewritten = rewriter.rewrite_for_retrieval(query)
        print(f"\nOriginal: {query}")
        print(f"Rewritten: {rewritten}")
    
    print("\n" + "="*60)
    print("STAGE 2: PROMPT REWRITING FOR RESPONSE")
    print("="*60)
    
    # Test case for Stage 2
    user_query = "How were the pyramids built?"
    retrieved_context = [
        "The Great Pyramid of Giza was built around 2560 BCE during the reign of Pharaoh Khufu. It took approximately 20 years to complete and required a workforce of around 100,000 workers.",
        "Ancient Egyptians used copper tools, wooden sledges, and ramps to move the massive limestone blocks. Each block weighed about 2.5 tons on average.",
        "Recent archaeological evidence suggests that the workers were not slaves but paid laborers who lived in nearby workers' villages."
    ]
    
    prompt = rewriter.rewrite_for_response(user_query, retrieved_context, language="en")
    print(f"\n{prompt}")
    
    print("\n" + "="*60)
    print("MULTILINGUAL RETRIEVAL")
    print("="*60)
    
    multilingual = rewriter.rewrite_for_multilingual_retrieval(
        "Tell me about the pyramids",
        target_languages=['en', 'ar']
    )
    for lang, query in multilingual.items():
        print(f"{lang.upper()}: {query}")
