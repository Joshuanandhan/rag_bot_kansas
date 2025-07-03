"""
Performance Testing Framework for RAG Agent
Tests multiple algorithms and configurations to optimize performance
"""

import time
import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Core imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import numpy as np

# For different search algorithms
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class PerformanceTester:
    def __init__(self, test_data_path: str = "output_all"):
        """
        Initialize the performance tester
        
        Args:
            test_data_path: Path to test data
        """
        self.test_data_path = test_data_path
        self.results = []
        self.test_queries = [
            "What are the requirements for a commercial driver's license in Kansas?",
            "How do I renew my driver's license?",
            "What documents do I need for vehicle registration?",
            "What are the penalties for DUI in Kansas?",
            "What is the speed limit on Kansas highways?",
            "How old do you have to be to get a motorcycle license?",
            "What are the vision requirements for driving?",
            "How much does it cost to get a Kansas ID card?",
            "What are the requirements for a learner's permit?",
            "How do I transfer my out-of-state license to Kansas?"
        ]
        
        # Configuration options to test
        self.embedding_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ]
        
        self.llm_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo"
        ]
        
        self.search_algorithms = [
            "faiss_l2",
            "faiss_cosine", 
            "basic_cosine"
        ]
        
        self.chunk_sizes = [3, 5, 7, 10]  # Number of chunks to retrieve
        
    def load_test_documents(self) -> List[Document]:
        """Load documents for testing"""
        try:
            from langchain_community.document_loaders import DirectoryLoader
            
            documents = []
            test_path = Path(self.test_data_path)
            
            if not test_path.exists():
                print(f"❌ Test data path {self.test_data_path} does not exist")
                return []
            
            # Load from all source directories
            for source_dir in test_path.iterdir():
                if source_dir.is_dir():
                    chunks_dir = source_dir / "chunks"
                    if chunks_dir.exists():
                        loader = DirectoryLoader(
                            str(chunks_dir),
                            glob="**/*.md"
                        )
                        docs = loader.load()
                        
                        # Add source metadata
                        for doc in docs:
                            doc.metadata['source_type'] = source_dir.name
                        
                        documents.extend(docs)
            
            print(f"📄 Loaded {len(documents)} test documents")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading test documents: {e}")
            return []
    
    def create_vector_store(self, documents: List[Document], embedding_model: str, 
                          search_algorithm: str) -> Optional[Any]:
        """Create vector store with specified configuration"""
        try:
            embeddings = OpenAIEmbeddings(model=embedding_model)
            
            if search_algorithm.startswith("faiss"):
                if search_algorithm == "faiss_l2":
                    # L2 distance (default)
                    vector_store = FAISS.from_documents(documents, embeddings)
                elif search_algorithm == "faiss_cosine":
                    # Cosine similarity
                    embedding_dim = len(embeddings.embed_query("test"))
                    index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine
                    vector_store = FAISS(
                        embedding_function=embeddings,
                        index=index,
                        docstore=faiss.swigfaiss.StandardGpuResources(),
                        index_to_docstore_id={}
                    )
                    vector_store.add_documents(documents)
                    
            elif search_algorithm == "basic_cosine":
                # Basic cosine similarity implementation
                vector_store = FAISS.from_documents(documents, embeddings)
                
            return vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            return None
    
    def test_retrieval_performance(self, vector_store: Any, query: str, 
                                 chunk_size: int) -> Dict[str, Any]:
        """Test retrieval performance"""
        start_time = time.time()
        
        try:
            results = vector_store.similarity_search(query, k=chunk_size)
            retrieval_time = time.time() - start_time
            
            return {
                "retrieval_time": retrieval_time,
                "chunks_retrieved": len(results),
                "success": True,
                "relevance_scores": [getattr(doc, 'score', 0.0) for doc in results[:3]]
            }
            
        except Exception as e:
            return {
                "retrieval_time": time.time() - start_time,
                "chunks_retrieved": 0,
                "success": False,
                "error": str(e),
                "relevance_scores": []
            }
    
    def test_generation_performance(self, llm_model: str, context: str, 
                                  query: str) -> Dict[str, Any]:
        """Test answer generation performance"""
        start_time = time.time()
        
        try:
            llm = ChatOpenAI(model=llm_model, temperature=0.7)
            
            prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
            
            response = llm.invoke(prompt)
            generation_time = time.time() - start_time
            
            return {
                "generation_time": generation_time,
                "response_length": len(response.content),
                "success": True,
                "response": response.content[:200] + "..." if len(response.content) > 200 else response.content
            }
            
        except Exception as e:
            return {
                "generation_time": time.time() - start_time,
                "response_length": 0,
                "success": False,
                "error": str(e),
                "response": ""
            }
    
    def run_single_test(self, embedding_model: str, search_algorithm: str, 
                       llm_model: str, chunk_size: int, query: str, 
                       vector_store: Any) -> Dict[str, Any]:
        """Run a single performance test"""
        
        print(f"   🔍 Testing: {embedding_model} | {search_algorithm} | {llm_model} | k={chunk_size}")
        
        # Test retrieval
        retrieval_results = self.test_retrieval_performance(vector_store, query, chunk_size)
        
        if not retrieval_results["success"]:
            return {
                "embedding_model": embedding_model,
                "search_algorithm": search_algorithm,
                "llm_model": llm_model,
                "chunk_size": chunk_size,
                "query": query,
                "success": False,
                "error": retrieval_results.get("error", "Unknown error"),
                **retrieval_results
            }
        
        # Get context for generation
        try:
            docs = vector_store.similarity_search(query, k=chunk_size)
            context = "\n\n".join([doc.page_content[:500] for doc in docs])
        except:
            context = "No context available"
        
        # Test generation
        generation_results = self.test_generation_performance(llm_model, context, query)
        
        # Calculate total time
        total_time = retrieval_results["retrieval_time"] + generation_results["generation_time"]
        
        return {
            "embedding_model": embedding_model,
            "search_algorithm": search_algorithm,
            "llm_model": llm_model,
            "chunk_size": chunk_size,
            "query": query,
            "retrieval_time": retrieval_results["retrieval_time"],
            "generation_time": generation_results["generation_time"],
            "total_time": total_time,
            "chunks_retrieved": retrieval_results["chunks_retrieved"],
            "response_length": generation_results["response_length"],
            "relevance_scores": retrieval_results["relevance_scores"],
            "response_preview": generation_results["response"],
            "success": retrieval_results["success"] and generation_results["success"]
        }
    
    def run_comprehensive_test(self, sample_size: int = 3) -> List[Dict[str, Any]]:
        """Run comprehensive performance testing"""
        
        print("🚀 Starting Comprehensive Performance Testing")
        print("=" * 60)
        
        # Load test documents
        documents = self.load_test_documents()
        if not documents:
            print("❌ No test documents available")
            return []
        
        results = []
        total_tests = (len(self.embedding_models) * len(self.search_algorithms) * 
                      len(self.llm_models) * len(self.chunk_sizes) * sample_size)
        test_count = 0
        
        # Test each configuration
        for embedding_model in self.embedding_models:
            print(f"\n📐 Testing embedding model: {embedding_model}")
            
            for search_algorithm in self.search_algorithms:
                print(f"  🔍 Testing search algorithm: {search_algorithm}")
                
                # Create vector store for this configuration
                vector_store = self.create_vector_store(documents, embedding_model, search_algorithm)
                if not vector_store:
                    print(f"  ❌ Failed to create vector store")
                    continue
                
                for llm_model in self.llm_models:
                    for chunk_size in self.chunk_sizes:
                        # Test with sample queries
                        for query in self.test_queries[:sample_size]:
                            test_count += 1
                            print(f"  📊 Progress: {test_count}/{total_tests}")
                            
                            result = self.run_single_test(
                                embedding_model, search_algorithm, llm_model,
                                chunk_size, query, vector_store
                            )
                            results.append(result)
                            
                            # Small delay to avoid rate limiting
                            time.sleep(0.1)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance test results"""
        
        if not results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful test results"}
        
        # Create DataFrame for analysis
        df = pd.DataFrame(successful_results)
        
        analysis = {
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "success_rate": len(successful_results) / len(results) * 100
            },
            "performance_metrics": {
                "avg_retrieval_time": df["retrieval_time"].mean(),
                "avg_generation_time": df["generation_time"].mean(),
                "avg_total_time": df["total_time"].mean(),
                "fastest_total_time": df["total_time"].min(),
                "slowest_total_time": df["total_time"].max()
            },
            "best_configurations": {
                "fastest_retrieval": df.loc[df["retrieval_time"].idxmin()].to_dict(),
                "fastest_generation": df.loc[df["generation_time"].idxmin()].to_dict(),
                "fastest_overall": df.loc[df["total_time"].idxmin()].to_dict()
            },
            "model_performance": {
                "by_embedding_model": df.groupby("embedding_model")["total_time"].agg(['mean', 'min', 'max']).to_dict(),
                "by_search_algorithm": df.groupby("search_algorithm")["total_time"].agg(['mean', 'min', 'max']).to_dict(),
                "by_llm_model": df.groupby("llm_model")["total_time"].agg(['mean', 'min', 'max']).to_dict(),
                "by_chunk_size": df.groupby("chunk_size")["total_time"].agg(['mean', 'min', 'max']).to_dict()
            }
        }
        
        return analysis
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], 
                    filename: str = None):
        """Save test results and analysis"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_{timestamp}"
        
        # Save raw results
        results_file = f"{filename}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save analysis
        analysis_file = f"{filename}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save CSV for easy viewing
        if results:
            df = pd.DataFrame([r for r in results if r.get("success", False)])
            csv_file = f"{filename}_results.csv"
            df.to_csv(csv_file, index=False)
        
        # Create summary report
        self.create_summary_report(analysis, f"{filename}_summary.md")
        
        print(f"\n✅ Results saved:")
        print(f"   📄 Raw results: {results_file}")
        print(f"   📊 Analysis: {analysis_file}")
        print(f"   📋 CSV: {csv_file}")
        print(f"   📝 Summary: {filename}_summary.md")
    
    def create_summary_report(self, analysis: Dict[str, Any], filename: str):
        """Create a markdown summary report"""
        
        with open(filename, 'w') as f:
            f.write("# RAG Agent Performance Test Results\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            # Summary
            if "summary" in analysis:
                f.write("## Test Summary\n\n")
                f.write(f"- **Total Tests:** {analysis['summary']['total_tests']}\n")
                f.write(f"- **Successful Tests:** {analysis['summary']['successful_tests']}\n")
                f.write(f"- **Success Rate:** {analysis['summary']['success_rate']:.1f}%\n\n")
            
            # Performance Metrics
            if "performance_metrics" in analysis:
                metrics = analysis['performance_metrics']
                f.write("## Performance Metrics\n\n")
                f.write(f"- **Average Retrieval Time:** {metrics['avg_retrieval_time']:.3f}s\n")
                f.write(f"- **Average Generation Time:** {metrics['avg_generation_time']:.3f}s\n")
                f.write(f"- **Average Total Time:** {metrics['avg_total_time']:.3f}s\n")
                f.write(f"- **Fastest Response:** {metrics['fastest_total_time']:.3f}s\n")
                f.write(f"- **Slowest Response:** {metrics['slowest_total_time']:.3f}s\n\n")
            
            # Best Configurations
            if "best_configurations" in analysis:
                f.write("## Top Performing Configurations\n\n")
                
                best_overall = analysis['best_configurations']['fastest_overall']
                f.write("### Fastest Overall Configuration\n")
                f.write(f"- **Embedding Model:** {best_overall['embedding_model']}\n")
                f.write(f"- **Search Algorithm:** {best_overall['search_algorithm']}\n")
                f.write(f"- **LLM Model:** {best_overall['llm_model']}\n")
                f.write(f"- **Chunk Size:** {best_overall['chunk_size']}\n")
                f.write(f"- **Total Time:** {best_overall['total_time']:.3f}s\n\n")
            
            # Model Comparisons
            if "model_performance" in analysis:
                f.write("## Model Performance Comparison\n\n")
                
                for category, data in analysis['model_performance'].items():
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    f.write("| Model | Avg Time (s) | Min Time (s) | Max Time (s) |\n")
                    f.write("|-------|--------------|--------------|---------------|\n")
                    
                    if isinstance(data, dict):
                        for model, stats in data.items():
                            if isinstance(stats, dict):
                                f.write(f"| {model} | {stats.get('mean', 0):.3f} | {stats.get('min', 0):.3f} | {stats.get('max', 0):.3f} |\n")
                    f.write("\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the test results:\n\n")
            f.write("1. **For fastest response times**, use the configuration listed above\n")
            f.write("2. **For cost optimization**, consider using smaller embedding models\n")
            f.write("3. **For accuracy**, experiment with different chunk sizes\n")
            f.write("4. **Monitor rate limits** when using OpenAI models in production\n")

def main():
    """Main function to run performance tests"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Agent Performance Tester")
    parser.add_argument("--data-path", type=str, default="output_all", 
                       help="Path to test data directory")
    parser.add_argument("--sample-size", type=int, default=3,
                       help="Number of queries to test per configuration")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename prefix")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PerformanceTester(args.data_path)
    
    print("🔬 RAG Agent Performance Testing Framework")
    print("=" * 50)
    print(f"📁 Test data path: {args.data_path}")
    print(f"📊 Sample size: {args.sample_size} queries per configuration")
    print(f"🧪 Total configurations: {len(tester.embedding_models) * len(tester.search_algorithms) * len(tester.llm_models) * len(tester.chunk_sizes)}")
    print("=" * 50)
    
    # Run tests
    try:
        results = tester.run_comprehensive_test(args.sample_size)
        
        if not results:
            print("❌ No test results generated")
            return
        
        # Analyze results
        analysis = tester.analyze_results(results)
        
        # Save results
        tester.save_results(results, analysis, args.output)
        
        # Print quick summary
        if "performance_metrics" in analysis:
            metrics = analysis["performance_metrics"]
            print(f"\n🎯 QUICK SUMMARY:")
            print(f"   Average response time: {metrics['avg_total_time']:.3f}s")
            print(f"   Fastest response: {metrics['fastest_total_time']:.3f}s")
            print(f"   Success rate: {analysis['summary']['success_rate']:.1f}%")
        
        if "best_configurations" in analysis:
            best = analysis["best_configurations"]["fastest_overall"]
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   {best['embedding_model']} + {best['search_algorithm']} + {best['llm_model']} (k={best['chunk_size']})")
            print(f"   Response time: {best['total_time']:.3f}s")
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    main() 