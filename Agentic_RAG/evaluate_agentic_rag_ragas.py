"""
RAGAS Evaluation for Agentic RAG System

This script evaluates the Agentic RAG system (with agents, tools, and multi-step reasoning)
using RAGAS metrics. It tests both the retrieval quality and the agentic workflow.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from Agentic_RAG folder
load_dotenv(Path(__file__).parent / ".env")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_similarity,
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Import agentic RAG components
from services.retriever_service import RetrieverService
from services.llm_service import LLMService
from services.query_rewriter_service import QueryRewriterService
from services.reranker_service import RerankerService


class AgenticRAGEvaluator:
    """Evaluates Agentic RAG system using RAGAS framework."""
    
    def __init__(self):
        """Initialize evaluator with RAG services."""
        print("🔧 Initializing Agentic RAG Evaluator...")
        
        # Check if Qdrant is running
        self._check_qdrant_connection()
        
        # Initialize services (same as your production system)
        try:
            print("   Loading retriever service...")
            self.retriever_service = RetrieverService()
            print("   ✓ Retriever service loaded")
        except Exception as e:
            print(f"   ✗ Retriever service failed: {e}")
            raise
        
        try:
            print("   Loading LLM service...")
            self.llm_service = LLMService()
            self.llm = self.llm_service.llm
            print("   ✓ LLM service loaded")
        except Exception as e:
            print(f"   ✗ LLM service failed: {e}")
            raise
        
        try:
            print("   Loading query rewriter...")
            self.query_rewriter = QueryRewriterService(llm_client=self.llm)
            print("   ✓ Query rewriter loaded")
        except Exception as e:
            print(f"   ✗ Query rewriter failed: {e}")
            raise
        
        try:
            print("   Loading reranker service...")
            self.reranker = RerankerService()
            print("   ✓ Reranker service loaded")
        except Exception as e:
            print(f"   ✗ Reranker service failed: {e}")
            raise
        
        # Configure embeddings for RAGAS
        try:
            print("   Loading embeddings for RAGAS...")
            embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.ragas_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            print("   ✓ Embeddings loaded")
        except Exception as e:
            print(f"   ✗ Embeddings failed: {e}")
            raise
        
        # Configure OpenRouter LLM for RAGAS metrics
        try:
            print("   Loading OpenRouter LLM for RAGAS...")
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            openrouter_model = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
            
            if not openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            
            self.ragas_llm = ChatOpenAI(
                model=openrouter_model,
                openai_api_key=openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
                max_tokens=1024,
                default_headers={
                    "HTTP-Referer": "https://github.com/AI-Tour-Guide",
                    "X-Title": "AI Tour Guide RAGAS Evaluation"
                }
            )
            print(f"   ✓ RAGAS LLM loaded: {openrouter_model}")
        except Exception as e:
            print(f"   ⚠️  RAGAS LLM failed: {e}")
            print("   ⚠️  Will use embedding-based metrics only")
            self.ragas_llm = None
        
        print("\n✅ Agentic RAG Evaluator initialized\n")
    
    def _check_qdrant_connection(self):
        """Check if Qdrant is running and accessible."""
        import requests
        
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        try:
            response = requests.get(f"{qdrant_url}/collections", timeout=5)
            if response.status_code == 200:
                print(f"✅ Qdrant is running at {qdrant_url}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Error: Cannot connect to Qdrant at {qdrant_url}")
            print(f"   {str(e)}\n")
            print("💡 Solution:")
            print("   1. Start Qdrant container:")
            print("      docker run -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant")
            print("\n   2. Or if already running, check the URL in your .env file")
            print(f"      QDRANT_URL={qdrant_url}\n")
            sys.exit(1)
    
    def generate_agentic_test_dataset(self) -> List[Dict[str, str]]:
        """
        Generate test dataset for agentic RAG (complex queries requiring multi-step reasoning).
        
        Returns:
            List of test cases with questions and ground truth
        """
        test_cases = [
            {
                "question": "Compare the architectural achievements of the Old Kingdom versus the New Kingdom in ancient Egypt",
                "ground_truth": "The Old Kingdom (2686-2181 BC) is famous for pyramid construction, particularly the Great Pyramids at Giza. The New Kingdom (1550-1069 BC) focused more on elaborate temples like Karnak and Luxor, and rock-cut tombs in the Valley of the Kings rather than pyramids."
            },
            {
                "question": "How did the Nile's flooding cycle influence both agriculture and religious beliefs in ancient Egypt?",
                "ground_truth": "The Nile's annual flooding deposited fertile silt for agriculture, dividing the year into three seasons. This cycle influenced religious beliefs, with gods like Hapi associated with the flood, and the concept of death and rebirth mirroring the agricultural cycle."
            },
            {
                "question": "What role did women play in ancient Egyptian society, and can you give examples of powerful female rulers?",
                "ground_truth": "Women in ancient Egypt had more rights than in many other ancient civilizations, including property ownership and legal rights. Powerful female rulers include Hatshepsut who ruled as pharaoh, Nefertiti who wielded significant political power, and Cleopatra VII who was the last pharaoh."
            },
            {
                "question": "Explain the connection between Egyptian hieroglyphics, the Rosetta Stone, and modern understanding of ancient Egypt",
                "ground_truth": "Hieroglyphics were the ancient Egyptian writing system. The Rosetta Stone, discovered in 1799, contained the same text in hieroglyphics, Demotic, and Greek. Jean-François Champollion used it to decipher hieroglyphics in 1822, unlocking the ability to read ancient Egyptian texts and vastly expanding modern understanding of the civilization."
            },
            {
                "question": "What were the key differences between the burial practices of common people versus pharaohs?",
                "ground_truth": "Pharaohs were buried in elaborate tombs (mastabas, pyramids, or rock-cut tombs) with extensive grave goods, golden artifacts, and complex mummification. Common people received simpler burials in shallow graves with basic mummification or just wrapped in linen, and minimal grave goods like pottery and food offerings."
            },
            {
                "question": "How did ancient Egypt's trade networks contribute to its wealth and power?",
                "ground_truth": "Egypt traded extensively via the Nile, Red Sea, and Mediterranean. They exported grain, papyrus, linen, and gold while importing cedar wood from Lebanon, copper from Cyprus, tin for bronze, incense from Punt, and luxury goods. This trade brought wealth, diplomatic ties, and cultural exchange that enhanced Egypt's power."
            },
            {
                "question": "Describe the evolution of Egyptian religious practices from polytheism through Akhenaten's reforms and back",
                "ground_truth": "Ancient Egypt was traditionally polytheistic with gods like Ra, Osiris, and Isis. Pharaoh Akhenaten (c. 1353-1336 BC) attempted monotheistic reform, promoting only Aten the sun disk. After his death, his son Tutankhamun restored traditional polytheism, and Akhenaten's reforms were reversed and largely erased."
            },
            {
                "question": "What technological and mathematical innovations did ancient Egyptians develop for pyramid construction?",
                "ground_truth": "Egyptians developed advanced surveying using merkhet and bay tools, rope stretching for right angles, copper and bronze tools, wooden sledges and lubricated sand for moving blocks, internal ramps, precise astronomical alignment, mathematical knowledge for calculating angles and volumes, and organizational systems for massive labor coordination."
            },
            # Additional test cases
            {
                "question": "Who was Tutankhamun and why is his tomb so famous?",
                "ground_truth": "Tutankhamun was a young pharaoh who ruled during the New Kingdom (1332-1323 BC). His tomb is famous because it was discovered nearly intact by Howard Carter in 1922, containing thousands of artifacts including his golden death mask, jewelry, chariots, and furniture, providing unprecedented insight into ancient Egyptian royal burial practices."
            },
            {
                "question": "What was the purpose of mummification in ancient Egypt?",
                "ground_truth": "Mummification was performed to preserve the body for the afterlife. Egyptians believed the soul (ka and ba) needed to return to the body after death. The process involved removing internal organs, drying the body with natron salt, wrapping in linen bandages, and placing in decorated coffins with amulets for protection."
            },
            {
                "question": "Describe the social hierarchy in ancient Egyptian society",
                "ground_truth": "Ancient Egyptian society was hierarchical with the pharaoh at the top as a living god. Below were priests and nobles, then scribes and government officials, followed by skilled craftsmen and merchants. Farmers formed the largest class, and at the bottom were servants and slaves who performed manual labor."
            },
            {
                "question": "What were the main gods and goddesses worshipped in ancient Egypt?",
                "ground_truth": "Major Egyptian deities included Ra the sun god, Osiris god of the underworld, Isis goddess of magic and motherhood, Horus the falcon-headed sky god, Anubis god of mummification, Thoth god of wisdom and writing, Hathor goddess of love, and Amun who became supreme deity during the New Kingdom."
            },
            {
                "question": "How was papyrus made and used in ancient Egypt?",
                "ground_truth": "Papyrus was made from the papyrus plant that grew along the Nile. The stem was cut into strips, laid in layers, pressed together, and dried to form sheets. It was used for writing religious texts, administrative records, literature, and letters. Scribes wrote with reed pens and ink made from soot or ochre."
            },
            {
                "question": "What was daily life like for ordinary Egyptians?",
                "ground_truth": "Ordinary Egyptians lived in mud-brick houses, worked primarily as farmers growing wheat and barley, and ate bread, beer, vegetables, and fish. Men typically worked in fields or as craftsmen while women managed households. Children learned their parents' trades. They wore linen clothing and celebrated religious festivals."
            },
            {
                "question": "Explain the significance of the Great Sphinx of Giza",
                "ground_truth": "The Great Sphinx is a limestone statue with a lion's body and human head, believed to represent Pharaoh Khafre. Built around 2500 BC during the Old Kingdom, it guards the Giza pyramid complex. It is one of the oldest and largest monumental sculptures in the world, symbolizing royal power and divine protection."
            },
            {
                "question": "What were the main crops and farming techniques in ancient Egypt?",
                "ground_truth": "Main crops included wheat and barley for bread and beer, flax for linen, and vegetables like onions, garlic, and lettuce. Farmers used the Nile's annual flooding to irrigate fields, employed shaduf devices to lift water, and used wooden plows pulled by cattle. They developed basin irrigation and canal systems."
            },
            {
                "question": "How did ancient Egyptians measure time and create calendars?",
                "ground_truth": "Egyptians created one of the first solar calendars with 365 days, divided into 12 months of 30 days plus 5 extra days. They divided the year into three seasons based on the Nile: Akhet (flooding), Peret (growing), and Shemu (harvest). They used sundials, water clocks (clepsydra), and star observations for timekeeping."
            },
            {
                "question": "What medical knowledge and practices existed in ancient Egypt?",
                "ground_truth": "Egyptian medicine combined practical treatments with magic and religion. They understood anatomy from mummification, used herbal remedies, performed surgery, set broken bones, and treated wounds with honey. Medical papyri like the Edwin Smith and Ebers papyri document diagnoses and treatments. Doctors (swnw) were respected professionals."
            },
            {
                "question": "Describe the construction and purpose of Egyptian temples",
                "ground_truth": "Egyptian temples were homes for gods, built with massive stone walls, towering pylons, columned halls, and inner sanctuaries. Priests performed daily rituals including offerings and prayers. Temples like Karnak and Luxor featured obelisks, colossal statues, and hieroglyphic inscriptions. They also served as economic centers with workshops and storerooms."
            },
            {
                "question": "What was the role of scribes in ancient Egyptian society?",
                "ground_truth": "Scribes were highly respected educated officials who could read and write hieroglyphics and hieratic script. They recorded taxes, kept government records, copied religious texts, and served as administrators. Training took years at temple schools. Being a scribe offered social mobility and exemption from manual labor and taxes."
            },
            {
                "question": "How did ancient Egypt interact with neighboring civilizations?",
                "ground_truth": "Egypt traded with Nubia for gold and ebony, Lebanon for cedar wood, Punt for incense, and Mediterranean peoples for various goods. They fought wars with Hittites, Libyans, and Sea Peoples. Diplomatic marriages and treaties maintained peace. Egypt influenced and was influenced by Greek, Mesopotamian, and African cultures."
            },
            {
                "question": "What caused the decline and fall of ancient Egyptian civilization?",
                "ground_truth": "Ancient Egypt declined due to multiple factors: invasions by Assyrians, Persians, and Greeks; internal political instability; economic problems; and climate changes affecting Nile flooding. Alexander the Great conquered Egypt in 332 BC, followed by Ptolemaic Greek rule. The civilization ended when Rome annexed Egypt in 30 BC after Cleopatra's death."
            }
        ]
        
        print(f"📝 Generated {len(test_cases)} agentic test cases\n")
        return test_cases
    
    def run_agentic_rag(self, question: str) -> Dict:
        """
        Run the full agentic RAG pipeline on a question.
        
        This mimics your production system:
        1. Query rewriting for retrieval optimization
        2. Hybrid search in Qdrant
        3. Reranking for precision
        4. LLM generation with context
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and contexts
        """
        # Step 1: Query rewriting (fallback to original if fails)
        try:
            retrieval_query = self.query_rewriter.rewrite_for_retrieval(question)
        except Exception as e:
            print(f"   ⚠️  Query rewriting failed, using original: {e}")
            retrieval_query = question
        
        # Step 2: Hybrid search (semantic + BM25)
        retriever = self.retriever_service.get_retriever(k=20, search_type="hybrid")
        docs = retriever.invoke(retrieval_query)
        
        # Step 3: Reranking to get top 3
        reranked_docs = self.reranker.rerank(retrieval_query, docs, top_k=3)
        contexts = [doc.page_content for doc in reranked_docs]
        
        # Step 4: Generate answer with LLM (fallback to context if fails)
        context_text = "\n\n".join(contexts)
        
        try:
            # Try LLM-based response generation
            prompt = self.query_rewriter.rewrite_for_response(
                user_query=question,
                retrieved_context=contexts,
                language="en"
            )
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            print(f"   ⚠️  LLM generation failed, using context summary: {e}")
            # Fallback: use first context as answer
            answer = f"Based on retrieved documents:\n\n{contexts[0][:500]}..."
        
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }
    
    def run_evaluation(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Run agentic RAG on all test questions.
        
        Args:
            test_cases: List of test cases with questions and ground truth
            
        Returns:
            List of results with questions, contexts, answers, and ground truth
        """
        results = []
        
        print("🤖 Running Agentic RAG pipeline on test questions...\n")
        print("Pipeline: Query Rewrite → Hybrid Search → Rerank → LLM Generation\n")
        
        for i, test_case in enumerate(test_cases, 1):
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]
            
            print(f"[{i}/{len(test_cases)}] {question[:70]}...")
            
            try:
                # Run full agentic RAG pipeline
                result = self.run_agentic_rag(question)
                
                print(f"   ✓ Answer generated ({len(result['answer'])} chars)")
                print(f"   ✓ Used {len(result['contexts'])} contexts\n")
                
                # Store in RAGAS format
                results.append({
                    "question": question,
                    "contexts": result["contexts"],
                    "answer": result["answer"],
                    "ground_truth": ground_truth
                })
                
            except Exception as e:
                print(f"   ✗ Error: {e}\n")
                results.append({
                    "question": question,
                    "contexts": ["Error"],
                    "answer": f"Error: {str(e)}",
                    "ground_truth": ground_truth
                })
        
        return results
    
    def evaluate_with_ragas(self, results: List[Dict]) -> Dict:
        """
        Evaluate results using RAGAS metrics (embedding-based only for free models).
        
        Args:
            results: List of RAG results
            
        Returns:
            Evaluation result object
        """
        print("\n" + "="*70)
        print("📊 RAGAS EVALUATION (Agentic RAG)")
        print("="*70 + "\n")
        
        # Convert to HuggingFace Dataset
        dataset_dict = {
            "question": [r["question"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "answer": [r["answer"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        print(f"Dataset size: {len(dataset)} examples\n")
        print("Evaluating with RAGAS metrics...")
        
        # Determine which metrics to use based on LLM availability
        if self.ragas_llm is not None:
            print("Using FULL metrics (LLM + Embedding-based):\n")
            print("  • faithfulness - Is the answer grounded in context?")
            print("  • answer_relevancy - Is the answer relevant to the question?")
            print("  • context_precision - Are the retrieved contexts precise?")
            print("  • context_recall - Do contexts cover the ground truth?")
            print("  • answer_correctness - Is the answer factually correct?")
            print("  • answer_similarity - Semantic similarity to ground truth\n")
            
            metrics_to_use = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity
            ]
            
            try:
                evaluation_result = evaluate(
                    dataset=dataset,
                    metrics=metrics_to_use,
                    llm=self.ragas_llm,
                    embeddings=self.ragas_embeddings,
                    raise_exceptions=False
                )
                
                print("✅ Evaluation complete!\n")
                return evaluation_result
                
            except Exception as e:
                print(f"⚠️  Full evaluation failed: {e}")
                print("Falling back to embedding-based metrics only...\n")
        
        # Fallback: Use only embedding-based metrics (no LLM required)
        print("Using EMBEDDING-BASED metrics only:\n")
        print("  • answer_similarity - Semantic similarity to ground truth\n")
        
        metrics_to_use = [
            answer_similarity
        ]
        
        try:
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                embeddings=self.ragas_embeddings,
                raise_exceptions=False
            )
            
            print("✅ Evaluation complete!\n")
            return evaluation_result
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_results(self, evaluation_result):
        """Print evaluation results."""
        if evaluation_result is None:
            print("No results to display.")
            return
        
        print("="*70)
        print("📈 AGENTIC RAG EVALUATION RESULTS")
        print("="*70 + "\n")
        
        # Debug: Show what we got
        print(f"Result type: {type(evaluation_result)}")
        
        # Define all possible metrics
        metric_names = [
            ("faithfulness", "Faithfulness", "How well the answer is grounded in context"),
            ("answer_relevancy", "Answer Relevancy", "How relevant the answer is to the question"),
            ("context_precision", "Context Precision", "Precision of retrieved contexts"),
            ("context_recall", "Context Recall", "Coverage of ground truth by contexts"),
            ("answer_correctness", "Answer Correctness", "Factual correctness of the answer"),
            ("answer_similarity", "Answer Similarity", "Semantic similarity to ground truth")
        ]
        
        print("\nRAGAS Metrics Summary:\n")
        
        scores_found = False
        
        # Try dictionary access first (RAGAS returns dict-like object)
        for attr_name, display_name, description in metric_names:
            score = None
            
            # Try dict-style access
            if hasattr(evaluation_result, '__getitem__'):
                try:
                    score = evaluation_result[attr_name]
                except (KeyError, TypeError):
                    pass
            
            # Try attribute access
            if score is None and hasattr(evaluation_result, attr_name):
                score = getattr(evaluation_result, attr_name)
            
            # Try .get() method
            if score is None and hasattr(evaluation_result, 'get'):
                score = evaluation_result.get(attr_name)
            
            if score is not None:
                try:
                    score_val = float(score)
                    if not pd.isna(score_val):
                        scores_found = True
                        bar = "█" * int(score_val * 20) + "░" * (20 - int(score_val * 20))
                        status = "✅" if score_val >= 0.7 else "⚠️" if score_val >= 0.5 else "❌"
                        print(f"  {status} {display_name:20} {score_val:.4f} [{bar}]")
                        print(f"      └─ {description}\n")
                except (ValueError, TypeError):
                    pass
        
        if not scores_found:
            print("  No metric scores available.")
            # Debug: Show what keys are available
            if hasattr(evaluation_result, 'keys'):
                print(f"\n  Available keys: {list(evaluation_result.keys())}")
            if hasattr(evaluation_result, '__dict__'):
                print(f"\n  Attributes: {list(evaluation_result.__dict__.keys())}")
        
        print("\n📖 Score Interpretation:")
        print("  ≥ 0.80 = Excellent | ≥ 0.70 = Good | ≥ 0.50 = Fair | < 0.50 = Needs Improvement")
        print("\n" + "="*70 + "\n")
    
    def save_results(self, evaluation_result, results: List[Dict], output_path: str = "agentic_ragas_results.csv"):
        """Save detailed results to CSV."""
        if evaluation_result is None:
            print("No results to save.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Add score if available
        if hasattr(evaluation_result, 'to_pandas'):
            scores_df = evaluation_result.to_pandas()
            for col in scores_df.columns:
                if col not in df.columns:
                    df[col] = scores_df[col]
        
        # Save
        output_file = Path(__file__).parent / output_path
        df.to_csv(output_file, index=False)
        
        print(f"💾 Results saved to: {output_file}\n")


def main():
    """Main evaluation workflow."""
    print("\n" + "="*70)
    print("🎯 AGENTIC RAG SYSTEM EVALUATION")
    print("="*70 + "\n")
    
    # Initialize evaluator
    evaluator = AgenticRAGEvaluator()
    
    # Generate test dataset (complex, multi-step questions)
    test_cases = evaluator.generate_agentic_test_dataset()
    
    # Run agentic RAG on all questions
    results = evaluator.run_evaluation(test_cases)
    
    # Evaluate with RAGAS
    evaluation_result = evaluator.evaluate_with_ragas(results)
    
    # Print and save results
    evaluator.print_results(evaluation_result)
    evaluator.save_results(evaluation_result, results)
    
    print("✅ Agentic RAG evaluation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
