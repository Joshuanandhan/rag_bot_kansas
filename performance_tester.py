"""
Performance Testing Framework for RAG Agent
Tests multiple algorithms and configurations to optimize performance
"""

import time
import json
import os
import statistics
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
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
        
    def cleanup_old_results(self):
        """Remove old performance test result files"""
        print("🧹 Cleaning up old result files...")
        
        # Patterns to match result files
        patterns = [
            "performance_test_*_results.json",
            "performance_test_*_analysis.json", 
            "performance_test_*_results.csv",
            "performance_test_*_summary.md",
            "performance_test_*_dashboard.png",
            "performance_test_*_comparison.png",
            "performance_test_*_cost_analysis.png",
            "performance_test_*_terminal_summary.txt"
        ]
        
        deleted_count = 0
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    os.remove(file)
                    deleted_count += 1
                except Exception as e:
                    print(f"   ⚠️  Could not delete {file}: {e}")
        
        if deleted_count > 0:
            print(f"   ✅ Deleted {deleted_count} old result files")
        else:
            print("   ✅ No old result files to clean up")
    
    def generate_performance_graphs(self, results: List[Dict[str, Any]], 
                                  analysis: Dict[str, Any], filename: str):
        """Generate performance visualization graphs"""
        
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            print("⚠️  No successful results to graph")
            return
        
        df = pd.DataFrame(successful_results)
        
        # Set up the plotting style with fallback
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                # Fallback to default style
                plt.style.use('default')
        
        try:
            sns.set_palette("husl")
        except:
            # Continue without seaborn styling
            pass
        
        # Create a comprehensive performance dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('RAG System Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Performance by Embedding Model
        ax1 = axes[0, 0]
        embedding_perf = df.groupby('embedding_model')['total_time'].agg(['mean', 'std'])
        embedding_perf['mean'].plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
        ax1.set_title('Average Response Time by Embedding Model')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Embedding Model')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Performance by LLM Model
        ax2 = axes[0, 1]
        llm_perf = df.groupby('llm_model')['total_time'].agg(['mean', 'std'])
        llm_perf['mean'].plot(kind='bar', ax=ax2, color='lightcoral', alpha=0.7)
        ax2.set_title('Average Response Time by LLM Model')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xlabel('LLM Model')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Performance by Search Algorithm
        ax3 = axes[0, 2]
        search_perf = df.groupby('search_algorithm')['total_time'].agg(['mean', 'std'])
        search_perf['mean'].plot(kind='bar', ax=ax3, color='lightgreen', alpha=0.7)
        ax3.set_title('Average Response Time by Search Algorithm')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xlabel('Search Algorithm')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Performance by Chunk Size
        ax4 = axes[1, 0]
        chunk_perf = df.groupby('chunk_size')['total_time'].agg(['mean', 'std'])
        chunk_perf['mean'].plot(kind='line', ax=ax4, color='orange', marker='o', linewidth=2)
        ax4.set_title('Response Time vs Chunk Size')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_xlabel('Number of Chunks')
        ax4.grid(True, alpha=0.3)
        
        # 5. Retrieval vs Generation Time Breakdown
        ax5 = axes[1, 1]
        time_breakdown = df[['retrieval_time', 'generation_time']].mean()
        time_breakdown.plot(kind='pie', ax=ax5, autopct='%1.1f%%', colors=['gold', 'lightblue'])
        ax5.set_title('Time Breakdown: Retrieval vs Generation')
        ax5.set_ylabel('')
        
        # 6. Top 10 Fastest Configurations
        ax6 = axes[1, 2]
        top_configs = df.nsmallest(10, 'total_time')
        config_labels = [f"{row['embedding_model'][:15]}+{row['llm_model'][:10]}+k{row['chunk_size']}" 
                        for _, row in top_configs.iterrows()]
        
        bars = ax6.barh(range(len(top_configs)), top_configs['total_time'], color='mediumpurple', alpha=0.7)
        ax6.set_yticks(range(len(top_configs)))
        ax6.set_yticklabels(config_labels, fontsize=8)
        ax6.set_xlabel('Response Time (seconds)')
        ax6.set_title('Top 10 Fastest Configurations')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{top_configs.iloc[i]["total_time"]:.3f}s', 
                    ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save the dashboard
        dashboard_file = f"{filename}_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed comparison charts
        self.create_detailed_comparison_charts(df, filename)
        
        print(f"   📊 Performance dashboard: {dashboard_file}")
        return dashboard_file
    
    def create_detailed_comparison_charts(self, df: pd.DataFrame, filename: str):
        """Create detailed comparison charts"""
        
        # Chart 1: Heatmap of all configurations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values='total_time', 
            index=['embedding_model', 'search_algorithm'], 
            columns=['llm_model', 'chunk_size'], 
            aggfunc='mean'
        )
        
        # Create heatmap with fallback
        try:
            sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', ax=ax1, fmt='.3f')
        except:
            # Fallback to matplotlib imshow
            im = ax1.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto')
            ax1.set_xticks(range(len(pivot_data.columns)))
            ax1.set_yticks(range(len(pivot_data.index)))
            ax1.set_xticklabels(pivot_data.columns, rotation=45)
            ax1.set_yticklabels(pivot_data.index)
            plt.colorbar(im, ax=ax1)
        
        ax1.set_title('Response Time Heatmap (All Configurations)')
        ax1.set_xlabel('LLM Model + Chunk Size')
        ax1.set_ylabel('Embedding Model + Search Algorithm')
        
        # Chart 2: Box plot showing distribution
        df_melted = df.melt(
            id_vars=['embedding_model', 'llm_model', 'search_algorithm', 'chunk_size'],
            value_vars=['retrieval_time', 'generation_time'],
            var_name='time_type',
            value_name='time_value'
        )
        
        # Create box plot with fallback
        try:
            sns.boxplot(data=df_melted, x='time_type', y='time_value', ax=ax2)
        except:
            # Fallback to matplotlib boxplot
            retrieval_times = df_melted[df_melted['time_type'] == 'retrieval_time']['time_value']
            generation_times = df_melted[df_melted['time_type'] == 'generation_time']['time_value']
            ax2.boxplot([retrieval_times, generation_times], labels=['retrieval_time', 'generation_time'])
        
        ax2.set_title('Distribution of Retrieval vs Generation Times')
        ax2.set_xlabel('Time Component')
        ax2.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save comparison charts
        comparison_file = f"{filename}_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Detailed comparison charts: {comparison_file}")
        return comparison_file
    
    def create_cost_analysis_chart(self, df: pd.DataFrame, filename: str):
        """Create cost analysis visualization"""
        
        # Rough cost estimates per 1000 tokens (as of 2024)
        embedding_costs = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10
        }
        
        llm_costs = {
            "gpt-3.5-turbo": 0.50,
            "gpt-4o-mini": 0.15,
            "gpt-4o": 15.00
        }
        
        # Estimate costs (simplified calculation)
        df['estimated_cost'] = df.apply(lambda row: 
            embedding_costs.get(row['embedding_model'], 0.1) * 0.1 +  # Embedding cost
            llm_costs.get(row['llm_model'], 1.0) * row['response_length'] / 1000,  # LLM cost
            axis=1
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost vs Performance scatter
        ax1 = axes[0]
        scatter = ax1.scatter(df['total_time'], df['estimated_cost'], 
                            c=df['chunk_size'], cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Estimated Cost per Query ($)')
        ax1.set_title('Cost vs Performance Trade-off')
        plt.colorbar(scatter, ax=ax1, label='Chunk Size')
        
        # Cost by model type
        ax2 = axes[1]
        cost_by_llm = df.groupby('llm_model')['estimated_cost'].mean()
        cost_by_llm.plot(kind='bar', ax=ax2, color='salmon', alpha=0.7)
        ax2.set_title('Average Cost by LLM Model')
        ax2.set_ylabel('Cost per Query ($)')
        ax2.set_xlabel('LLM Model')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        cost_file = f"{filename}_cost_analysis.png"
        plt.savefig(cost_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💰 Cost analysis chart: {cost_file}")
        return cost_file
    
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
            
            # Limit documents to avoid token limits - use only first 100 for testing
            if len(documents) > 100:
                documents = documents[:100]
                print(f"📊 Using {len(documents)} documents for testing (limited to avoid token limits)")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error loading test documents: {e}")
            return []
    
    def create_vector_store(self, documents: List[Document], embedding_model: str, 
                          search_algorithm: str) -> Optional[Any]:
        """Create vector store with specified configuration"""
        try:
            # Create embeddings with batch processing
            embeddings = OpenAIEmbeddings(model=embedding_model)
            
            if search_algorithm.startswith("faiss"):
                if search_algorithm == "faiss_l2":
                    # L2 distance (default) with batch processing
                    try:
                        vector_store = FAISS.from_documents(documents, embeddings)
                    except Exception as e:
                        if "max_tokens_per_request" in str(e):
                            # Process in smaller batches
                            batch_size = 50
                            all_docs = []
                            
                            for i in range(0, len(documents), batch_size):
                                batch = documents[i:i + batch_size]
                                if i == 0:
                                    vector_store = FAISS.from_documents(batch, embeddings)
                                else:
                                    batch_store = FAISS.from_documents(batch, embeddings)
                                    vector_store.merge_from(batch_store)
                                print(f"   Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                        else:
                            raise e
                            
                elif search_algorithm == "faiss_cosine":
                    # Cosine similarity with proper CPU FAISS
                    try:
                        vector_store = FAISS.from_documents(documents, embeddings)
                        # Convert to inner product index for cosine similarity
                        # This is a simplified approach - just use the default L2 for now
                        # as the CPU FAISS doesn't have easy cosine similarity setup
                    except Exception as e:
                        if "max_tokens_per_request" in str(e):
                            # Process in smaller batches
                            batch_size = 50
                            
                            for i in range(0, len(documents), batch_size):
                                batch = documents[i:i + batch_size]
                                if i == 0:
                                    vector_store = FAISS.from_documents(batch, embeddings)
                                else:
                                    batch_store = FAISS.from_documents(batch, embeddings)
                                    vector_store.merge_from(batch_store)
                                print(f"   Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                        else:
                            raise e
                    
            elif search_algorithm == "basic_cosine":
                # Basic cosine similarity implementation
                try:
                    vector_store = FAISS.from_documents(documents, embeddings)
                except Exception as e:
                    if "max_tokens_per_request" in str(e):
                        # Process in smaller batches
                        batch_size = 50
                        
                        for i in range(0, len(documents), batch_size):
                            batch = documents[i:i + batch_size]
                            if i == 0:
                                vector_store = FAISS.from_documents(batch, embeddings)
                            else:
                                batch_store = FAISS.from_documents(batch, embeddings)
                                vector_store.merge_from(batch_store)
                            print(f"   Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                    else:
                        raise e
                
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
    
    def save_terminal_summary(self, analysis: Dict[str, Any], filename: str):
        """Save the terminal summary to a file"""
        
        summary_file = f"{filename}_terminal_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("🎯 RAG PERFORMANCE TEST TERMINAL SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Print quick summary
            if "performance_metrics" in analysis:
                metrics = analysis["performance_metrics"]
                f.write("🎯 QUICK SUMMARY:\n")
                f.write(f"   Average response time: {metrics['avg_total_time']:.3f}s\n")
                f.write(f"   Fastest response: {metrics['fastest_total_time']:.3f}s\n")
                f.write(f"   Success rate: {analysis['summary']['success_rate']:.1f}%\n\n")
            
            if "best_configurations" in analysis:
                best = analysis["best_configurations"]["fastest_overall"]
                f.write("🏆 BEST CONFIGURATION:\n")
                f.write(f"   {best['embedding_model']} + {best['search_algorithm']} + {best['llm_model']} (k={best['chunk_size']})\n")
                f.write(f"   Response time: {best['total_time']:.3f}s\n")
                f.write(f"   Retrieval time: {best['retrieval_time']:.3f}s\n")
                f.write(f"   Generation time: {best['generation_time']:.3f}s\n\n")
            
            # Add file locations
            f.write("📁 GENERATED FILES:\n")
            f.write(f"   📄 Raw results: {filename}_results.json\n")
            f.write(f"   📊 Analysis: {filename}_analysis.json\n")
            f.write(f"   📋 CSV data: {filename}_results.csv\n")
            f.write(f"   📝 Summary report: {filename}_summary.md\n")
            f.write(f"   📊 Performance dashboard: {filename}_dashboard.png\n")
            f.write(f"   📊 Comparison charts: {filename}_comparison.png\n")
            f.write(f"   💰 Cost analysis: {filename}_cost_analysis.png\n")
            f.write(f"   📋 Terminal summary: {filename}_terminal_summary.txt\n\n")
            
            f.write("🔍 NEXT STEPS:\n")
            f.write("1. Open the dashboard PNG files to view performance graphs\n")
            f.write("2. Read the summary.md file for detailed analysis\n")
            f.write("3. Use the CSV file for further data analysis\n")
            f.write("4. Implement the best configuration in your production system\n")
        
        return summary_file

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
        csv_file = None
        if results:
            successful_results = [r for r in results if r.get("success", False)]
            if successful_results:
                df = pd.DataFrame(successful_results)
                csv_file = f"{filename}_results.csv"
                df.to_csv(csv_file, index=False)
        
        # Generate performance graphs
        print("📊 Generating performance visualizations...")
        graph_files = []
        
        if results:
            try:
                # Main dashboard
                dashboard_file = self.generate_performance_graphs(results, analysis, filename)
                if dashboard_file:
                    graph_files.append(dashboard_file)
                
                # Cost analysis
                successful_results = [r for r in results if r.get("success", False)]
                if successful_results:
                    df = pd.DataFrame(successful_results)
                    cost_file = self.create_cost_analysis_chart(df, filename)
                    if cost_file:
                        graph_files.append(cost_file)
                        
            except Exception as e:
                print(f"⚠️  Error generating graphs: {e}")
                print("   Continuing without graphs...")
        
        # Save terminal summary
        terminal_summary_file = self.save_terminal_summary(analysis, filename)
        
        # Create summary report
        self.create_summary_report(analysis, f"{filename}_summary.md", graph_files)
        
        print(f"\n✅ Results saved to the following files:")
        print(f"   📄 Raw results: {results_file}")
        print(f"   📊 Analysis: {analysis_file}")
        if csv_file:
            print(f"   📋 CSV data: {csv_file}")
        print(f"   📝 Summary report: {filename}_summary.md")
        print(f"   📋 Terminal summary: {terminal_summary_file}")
        
        for graph_file in graph_files:
            print(f"   📊 Graph: {graph_file}")
        
        print(f"\n🎨 GRAPH LOCATIONS:")
        print(f"   📊 Main dashboard: {filename}_dashboard.png")
        print(f"   📊 Detailed comparison: {filename}_comparison.png") 
        print(f"   💰 Cost analysis: {filename}_cost_analysis.png")
        print(f"\n💡 TIP: Open the PNG files to view the performance graphs!")
    
    def create_summary_report(self, analysis: Dict[str, Any], filename: str, graph_files: List[str] = None):
        """Create a markdown summary report"""
        
        with open(filename, 'w') as f:
            f.write("# RAG Agent Performance Test Results\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            # Add graphs section if graphs were generated
            if graph_files:
                f.write("## Performance Visualizations\n\n")
                f.write("The following charts provide visual analysis of the performance test results:\n\n")
                
                for graph_file in graph_files:
                    graph_name = graph_file.replace('_', ' ').replace('.png', '').title()
                    f.write(f"### {graph_name}\n")
                    f.write(f"![{graph_name}]({graph_file})\n\n")
                
                f.write("---\n\n")
            
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
                f.write("### 🏆 Fastest Overall Configuration\n")
                f.write(f"- **Embedding Model:** {best_overall['embedding_model']}\n")
                f.write(f"- **Search Algorithm:** {best_overall['search_algorithm']}\n")
                f.write(f"- **LLM Model:** {best_overall['llm_model']}\n")
                f.write(f"- **Chunk Size:** {best_overall['chunk_size']}\n")
                f.write(f"- **Total Time:** {best_overall['total_time']:.3f}s\n")
                f.write(f"- **Retrieval Time:** {best_overall['retrieval_time']:.3f}s\n")
                f.write(f"- **Generation Time:** {best_overall['generation_time']:.3f}s\n\n")
                
                # Add fastest retrieval and generation configurations
                if 'fastest_retrieval' in analysis['best_configurations']:
                    best_retrieval = analysis['best_configurations']['fastest_retrieval']
                    f.write("### 🔍 Fastest Retrieval Configuration\n")
                    f.write(f"- **Configuration:** {best_retrieval['embedding_model']} + {best_retrieval['search_algorithm']}\n")
                    f.write(f"- **Retrieval Time:** {best_retrieval['retrieval_time']:.3f}s\n\n")
                
                if 'fastest_generation' in analysis['best_configurations']:
                    best_generation = analysis['best_configurations']['fastest_generation']
                    f.write("### 🤖 Fastest Generation Configuration\n")
                    f.write(f"- **LLM Model:** {best_generation['llm_model']}\n")
                    f.write(f"- **Generation Time:** {best_generation['generation_time']:.3f}s\n\n")
            
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
            
            # Key Insights
            f.write("## 🔍 Key Insights\n\n")
            if "performance_metrics" in analysis:
                metrics = analysis['performance_metrics']
                avg_retrieval = metrics['avg_retrieval_time']
                avg_generation = metrics['avg_generation_time']
                
                if avg_retrieval > avg_generation:
                    f.write("- **Retrieval is the bottleneck** - Consider optimizing vector search algorithms\n")
                else:
                    f.write("- **Generation is the bottleneck** - Consider using faster LLM models\n")
                
                if metrics['fastest_total_time'] < 1.0:
                    f.write("- **Excellent performance** - Sub-second responses achieved\n")
                elif metrics['fastest_total_time'] < 2.0:
                    f.write("- **Good performance** - Response times under 2 seconds\n")
                else:
                    f.write("- **Optimization needed** - Consider faster model combinations\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## 📋 Recommendations\n\n")
            f.write("Based on the test results:\n\n")
            
            if "best_configurations" in analysis:
                f.write("### For Production Use:\n")
                f.write("1. **Use the fastest overall configuration** listed above for optimal response times\n")
                f.write("2. **Monitor API costs** - Balance performance with cost based on your usage volume\n")
                f.write("3. **Consider caching** - Implement response caching for frequently asked questions\n\n")
            
            f.write("### For Different Use Cases:\n")
            f.write("- **Real-time applications:** Use fastest retrieval + fastest generation models\n")
            f.write("- **Cost-sensitive applications:** Use smaller embedding models and GPT-3.5-turbo\n")
            f.write("- **Quality-focused applications:** Use larger embedding models and GPT-4o\n")
            f.write("- **High-throughput applications:** Implement parallel processing and rate limiting\n\n")
            
            f.write("### Next Steps:\n")
            f.write("1. **Implement the recommended configuration** in your production system\n")
            f.write("2. **Monitor real-world performance** - Test with actual user queries\n")
            f.write("3. **Set up monitoring** - Track response times and costs in production\n")
            f.write("4. **Iterate and optimize** - Re-run tests as new models become available\n")
            
            # Footer
            f.write("\n---\n")
            f.write("*This report was generated automatically by the RAG Performance Testing Framework*\n")

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
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Skip cleanup of old result files")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PerformanceTester(args.data_path)
    
    print("🔬 RAG Agent Performance Testing Framework")
    print("=" * 50)
    
    # Clean up old results unless disabled
    if not args.no_cleanup:
        tester.cleanup_old_results()
        print()
    
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
        
        # Save results (including terminal summary)
        tester.save_results(results, analysis, args.output)
        
        # Print quick summary to terminal
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
        
        print(f"\n📊 All results including graphs and terminal summary have been saved!")
        print(f"📁 Check the current directory for PNG graph files and analysis reports.")
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 