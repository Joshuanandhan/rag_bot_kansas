"""
Vector Search Benchmark
Tests different vector search algorithms and indexing strategies
"""

import time
import glob
import os
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class VectorBenchmark:
    def __init__(self):
        self.test_queries = [
            "commercial driver license requirements Kansas",
            "motorcycle safety gear protection",
            "vehicle registration documents needed",
            "DUI penalties Kansas driving",
            "speed limit highway Kansas"
        ]
        
        # Different FAISS index types to test
        self.index_types = {
            "Flat L2": "flat_l2",
            "Flat IP": "flat_ip", 
            "IVF Flat": "ivf_flat",
            "IVF PQ": "ivf_pq",
            "HNSW": "hnsw"
        }
    
    def cleanup_old_results(self):
        """Remove old vector benchmark result files"""
        print("🧹 Cleaning up old vector benchmark files...")
        
        # Patterns to match result files
        patterns = [
            "vector_benchmark_*.json",
            "vector_benchmark_*_dashboard.png",
            "vector_benchmark_*_comparison.png", 
            "vector_benchmark_*_terminal_summary.txt"
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
    
    def generate_vector_benchmark_graphs(self, results: List[Dict], filename: str):
        """Generate graphs for vector benchmark results"""
        
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            print("⚠️  No successful results to graph")
            return []
        
        df = pd.DataFrame(successful_results)
        
        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Vector Search Benchmark Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Search Speed Comparison
        ax1 = axes[0, 0]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        bars = ax1.bar(df['index_name'], df['avg_search_time_ms'], color=colors[:len(df)], alpha=0.8)
        ax1.set_title('Average Search Time by Index Type')
        ax1.set_ylabel('Search Time (ms)')
        ax1.set_xlabel('Index Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 2. Index Creation Time
        ax2 = axes[0, 1]
        bars = ax2.bar(df['index_name'], df['creation_time'], color=colors[:len(df)], alpha=0.8)
        ax2.set_title('Index Creation Time')
        ax2.set_ylabel('Creation Time (seconds)')
        ax2.set_xlabel('Index Type')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # 3. Memory Usage
        ax3 = axes[1, 0]
        bars = ax3.bar(df['index_name'], df['memory_usage_mb'], color=colors[:len(df)], alpha=0.8)
        ax3.set_title('Memory Usage')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_xlabel('Index Type')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}MB', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance vs Memory Trade-off
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['avg_search_time_ms'], df['memory_usage_mb'], 
                            c=colors[:len(df)], s=100, alpha=0.8)
        
        # Add labels for each point
        for i, (_, row) in enumerate(df.iterrows()):
            ax4.annotate(row['index_name'], 
                        (row['avg_search_time_ms'], row['memory_usage_mb']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left')
        
        ax4.set_title('Performance vs Memory Trade-off')
        ax4.set_xlabel('Average Search Time (ms)')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = f"{filename}_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Vector benchmark dashboard: {dashboard_file}")
        return [dashboard_file]
    
    def save_terminal_summary(self, results: List[Dict], filename: str):
        """Save terminal summary for vector benchmark"""
        
        summary_file = f"{filename}_terminal_summary.txt"
        successful_results = [r for r in results if r.get("success", False)]
        
        with open(summary_file, 'w') as f:
            f.write("🎯 VECTOR SEARCH BENCHMARK TERMINAL SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            if successful_results:
                # Sort by search speed
                by_search_speed = sorted(successful_results, key=lambda x: x["avg_search_time_ms"])
                
                f.write("🏆 RANKING BY SEARCH SPEED:\n")
                for i, result in enumerate(by_search_speed, 1):
                    f.write(f"{i}. {result['index_name']}: {result['avg_search_time_ms']:.2f}ms avg\n")
                
                # Sort by creation time
                by_creation_time = sorted(successful_results, key=lambda x: x["creation_time"])
                
                f.write(f"\n⚡ RANKING BY INDEX CREATION SPEED:\n")
                for i, result in enumerate(by_creation_time, 1):
                    f.write(f"{i}. {result['index_name']}: {result['creation_time']:.3f}s\n")
                
                # Sort by memory usage
                by_memory = sorted(successful_results, key=lambda x: x["memory_usage_mb"])
                
                f.write(f"\n💾 RANKING BY MEMORY USAGE:\n")
                for i, result in enumerate(by_memory, 1):
                    f.write(f"{i}. {result['index_name']}: {result['memory_usage_mb']:.1f} MB\n")
                
                # Recommendations
                fastest_search = by_search_speed[0]
                fastest_creation = by_creation_time[0]
                lowest_memory = by_memory[0]
                
                f.write(f"\n🎯 RECOMMENDATIONS:\n")
                f.write(f"⚡ Fastest Search: {fastest_search['index_name']} ({fastest_search['avg_search_time_ms']:.2f}ms)\n")
                f.write(f"🚀 Fastest Creation: {fastest_creation['index_name']} ({fastest_creation['creation_time']:.3f}s)\n")
                f.write(f"💾 Lowest Memory: {lowest_memory['index_name']} ({lowest_memory['memory_usage_mb']:.1f}MB)\n")
                
                # Overall recommendation
                if len(successful_results) > 0:
                    # Calculate weighted scores
                    for result in successful_results:
                        search_rank = by_search_speed.index(result) + 1
                        creation_rank = by_creation_time.index(result) + 1
                        memory_rank = by_memory.index(result) + 1
                        
                        weighted_score = (search_rank * 0.5 + creation_rank * 0.3 + memory_rank * 0.2)
                        result['overall_score'] = weighted_score
                    
                    best_overall = min(successful_results, key=lambda x: x['overall_score'])
                    f.write(f"\n🏆 BEST OVERALL: {best_overall['index_name']}\n")
                    f.write(f"   Search: {best_overall['avg_search_time_ms']:.2f}ms\n")
                    f.write(f"   Creation: {best_overall['creation_time']:.3f}s\n")
                    f.write(f"   Memory: {best_overall['memory_usage_mb']:.1f}MB\n")
            
            f.write(f"\n📁 GENERATED FILES:\n")
            f.write(f"   📄 Raw results: {filename}.json\n")
            f.write(f"   📊 Dashboard: {filename}_dashboard.png\n")
            f.write(f"   📋 Terminal summary: {filename}_terminal_summary.txt\n")
            
            f.write(f"\n🔍 NEXT STEPS:\n")
            f.write("1. View the dashboard PNG file for visual analysis\n")
            f.write("2. Use the best overall index type for your use case\n")
            f.write("3. Consider the trade-offs between speed, memory, and creation time\n")
            f.write("4. Test with your actual data size and query patterns\n")
        
        return summary_file

    def load_documents_and_embeddings(self):
        """Load documents and create embeddings"""
        print("📄 Loading documents and creating embeddings...")
        
        documents = []
        output_path = Path("output_all")
        
        if not output_path.exists():
            print("❌ No output_all directory found")
            return [], []
        
        # Load documents
        for source_dir in output_path.iterdir():
            if source_dir.is_dir():
                chunks_dir = source_dir / "chunks"
                if chunks_dir.exists():
                    loader = DirectoryLoader(str(chunks_dir), glob="**/*.md")
                    docs = loader.load()
                    documents.extend(docs)
        
        if not documents:
            print("❌ No documents found")
            return [], []
        
        print(f"📊 Found {len(documents)} documents")
        
        # For benchmark, use a reasonable sample to avoid token limits
        # Use first 50 documents for testing
        sample_size = min(50, len(documents))
        documents = documents[:sample_size]
        print(f"📊 Using {sample_size} documents for testing")
        
        # Create embeddings
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")  # Use small for speed
        
        print("🔄 Creating embeddings...")
        texts = [doc.page_content for doc in documents]
        
        # Process in batches to avoid token limits
        batch_size = 100  # Process 100 documents at a time
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch_texts)} docs)")
            
            try:
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"   ❌ Error processing batch: {e}")
                # Try smaller batch if it fails
                if len(batch_texts) > 1:
                    print(f"   🔄 Retrying with smaller batches...")
                    for single_text in batch_texts:
                        try:
                            single_embedding = embeddings_model.embed_documents([single_text])
                            all_embeddings.extend(single_embedding)
                        except Exception as e2:
                            print(f"   ❌ Failed to process single document: {e2}")
                            continue
        
        if not all_embeddings:
            print("❌ No embeddings created")
            return [], []
        
        print(f"✅ Created {len(all_embeddings)} embeddings of dimension {len(all_embeddings[0])}")
        
        # Truncate documents to match embeddings
        documents = documents[:len(all_embeddings)]
        
        return documents, np.array(all_embeddings)
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str) -> faiss.Index:
        """Create different types of FAISS indexes"""
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        print(f"  🔧 Creating {index_type} index for {n_vectors} vectors of dim {dimension}")
        
        if index_type == "flat_l2":
            # Flat L2 (exact search)
            index = faiss.IndexFlatL2(dimension)
            
        elif index_type == "flat_ip":
            # Flat Inner Product (cosine similarity)
            index = faiss.IndexFlatIP(dimension)
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
        elif index_type == "ivf_flat":
            # IVF with flat vectors (approximate search)
            nlist = min(100, n_vectors // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
        elif index_type == "ivf_pq":
            # IVF with Product Quantization (memory efficient)
            nlist = min(100, n_vectors // 10)
            m = 8  # Number of subquantizers
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            
        elif index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World)
            M = 16  # Number of bi-directional links
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 200
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return index
    
    def benchmark_index_creation(self, embeddings: np.ndarray, index_type: str) -> Dict[str, Any]:
        """Benchmark index creation time"""
        
        start_time = time.time()
        
        try:
            index = self.create_faiss_index(embeddings, index_type)
            
            # Train index if needed
            if hasattr(index, 'is_trained') and not index.is_trained:
                print(f"    🎯 Training {index_type} index...")
                train_start = time.time()
                index.train(embeddings)
                train_time = time.time() - train_start
            else:
                train_time = 0
            
            # Add vectors
            add_start = time.time()
            index.add(embeddings)
            add_time = time.time() - add_start
            
            creation_time = time.time() - start_time
            
            return {
                "success": True,
                "creation_time": creation_time,
                "train_time": train_time,
                "add_time": add_time,
                "index": index,
                "memory_usage": index.ntotal * embeddings.shape[1] * 4 / (1024 * 1024)  # Rough MB estimate
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "creation_time": time.time() - start_time
            }
    
    def benchmark_search(self, index: faiss.Index, query_embeddings: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """Benchmark search performance"""
        
        search_times = []
        
        for query_emb in query_embeddings:
            start_time = time.time()
            
            try:
                # Reshape for single query
                query_vector = query_emb.reshape(1, -1)
                
                # Search
                distances, indices = index.search(query_vector, k)
                
                search_time = time.time() - start_time
                search_times.append(search_time)
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "success": True,
            "avg_search_time": np.mean(search_times),
            "min_search_time": np.min(search_times),
            "max_search_time": np.max(search_times),
            "std_search_time": np.std(search_times),
            "total_queries": len(search_times)
        }
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive vector search benchmark"""
        
        print("🚀 Vector Search Benchmark")
        print("=" * 50)
        
        # Clean up old results
        self.cleanup_old_results()
        print()
        
        # Load data
        documents, embeddings = self.load_documents_and_embeddings()
        if len(documents) == 0:
            return
        
        # Create query embeddings
        print("🔍 Creating query embeddings...")
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        query_embeddings = np.array(embeddings_model.embed_documents(self.test_queries))
        
        results = []
        
        # Test each index type
        for index_name, index_type in self.index_types.items():
            print(f"\n📊 Testing {index_name} ({index_type})")
            print("-" * 30)
            
            # Benchmark index creation
            creation_result = self.benchmark_index_creation(embeddings, index_type)
            
            if not creation_result["success"]:
                print(f"  ❌ Failed to create index: {creation_result['error']}")
                results.append({
                    "index_name": index_name,
                    "index_type": index_type,
                    "success": False,
                    "error": creation_result["error"]
                })
                continue
            
            print(f"  ✅ Index created in {creation_result['creation_time']:.3f}s")
            if creation_result['train_time'] > 0:
                print(f"     🎯 Training time: {creation_result['train_time']:.3f}s")
            print(f"     📦 Add time: {creation_result['add_time']:.3f}s")
            print(f"     💾 Est. memory: {creation_result['memory_usage']:.1f} MB")
            
            # Benchmark search
            search_result = self.benchmark_search(creation_result["index"], query_embeddings)
            
            if not search_result["success"]:
                print(f"  ❌ Search failed: {search_result['error']}")
                results.append({
                    "index_name": index_name,
                    "index_type": index_type,
                    "success": False,
                    "error": search_result["error"]
                })
                continue
            
            print(f"  🔍 Average search time: {search_result['avg_search_time']*1000:.2f}ms")
            print(f"     ⚡ Min: {search_result['min_search_time']*1000:.2f}ms | Max: {search_result['max_search_time']*1000:.2f}ms")
            
            # Combine results
            combined_result = {
                "index_name": index_name,
                "index_type": index_type,
                "success": True,
                "creation_time": creation_result["creation_time"],
                "train_time": creation_result["train_time"],
                "add_time": creation_result["add_time"],
                "memory_usage_mb": creation_result["memory_usage"],
                "avg_search_time_ms": search_result["avg_search_time"] * 1000,
                "min_search_time_ms": search_result["min_search_time"] * 1000,
                "max_search_time_ms": search_result["max_search_time"] * 1000,
                "std_search_time_ms": search_result["std_search_time"] * 1000,
                "num_vectors": embeddings.shape[0],
                "vector_dimension": embeddings.shape[1]
            }
            
            results.append(combined_result)
        
        # Analysis and summary
        self.analyze_results(results)
        
        # Save results and generate graphs
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"vector_benchmark_{timestamp}.json"
        filename_base = f"vector_benchmark_{timestamp}"
        
        # Save JSON results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate graphs
        print("\n📊 Generating performance visualizations...")
        try:
            graph_files = self.generate_vector_benchmark_graphs(results, filename_base)
            terminal_summary = self.save_terminal_summary(results, filename_base)
            
            print(f"\n✅ Vector benchmark completed! Files saved:")
            print(f"   📄 Raw results: {results_file}")
            print(f"   📊 Dashboard: {filename_base}_dashboard.png")
            print(f"   📋 Terminal summary: {terminal_summary}")
            print(f"\n💡 TIP: Open the PNG file to view the performance dashboard!")
            
        except Exception as e:
            print(f"⚠️  Error generating graphs: {e}")
            print(f"   📄 Raw results saved to: {results_file}")
            print("   Continuing without graphs...")
    
    def analyze_results(self, results: List[Dict[str, Any]]):
        """Analyze and display benchmark results"""
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            print("\n❌ No successful benchmark results")
            return
        
        print(f"\n📊 BENCHMARK SUMMARY")
        print("=" * 50)
        
        # Sort by search speed
        by_search_speed = sorted(successful_results, key=lambda x: x["avg_search_time_ms"])
        
        print("🏆 Ranking by Search Speed:")
        for i, result in enumerate(by_search_speed, 1):
            print(f"{i}. {result['index_name']}: {result['avg_search_time_ms']:.2f}ms avg")
        
        # Sort by creation time
        by_creation_time = sorted(successful_results, key=lambda x: x["creation_time"])
        
        print(f"\n⚡ Ranking by Index Creation Speed:")
        for i, result in enumerate(by_creation_time, 1):
            print(f"{i}. {result['index_name']}: {result['creation_time']:.3f}s")
        
        # Sort by memory usage
        by_memory = sorted(successful_results, key=lambda x: x["memory_usage_mb"])
        
        print(f"\n💾 Ranking by Memory Usage:")
        for i, result in enumerate(by_memory, 1):
            print(f"{i}. {result['index_name']}: {result['memory_usage_mb']:.1f} MB")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        print("-" * 30)
        
        fastest_search = by_search_speed[0]
        fastest_creation = by_creation_time[0]
        lowest_memory = by_memory[0]
        
        print(f"⚡ Fastest Search: {fastest_search['index_name']} ({fastest_search['avg_search_time_ms']:.2f}ms)")
        print(f"🚀 Fastest Creation: {fastest_creation['index_name']} ({fastest_creation['creation_time']:.3f}s)")
        print(f"💾 Lowest Memory: {lowest_memory['index_name']} ({lowest_memory['memory_usage_mb']:.1f}MB)")
        
        # Overall recommendation
        if len(successful_results) > 0:
            # Weight: 50% search speed, 30% creation time, 20% memory
            for result in successful_results:
                search_rank = by_search_speed.index(result) + 1
                creation_rank = by_creation_time.index(result) + 1
                memory_rank = by_memory.index(result) + 1
                
                weighted_score = (search_rank * 0.5 + creation_rank * 0.3 + memory_rank * 0.2)
                result['overall_score'] = weighted_score
            
            best_overall = min(successful_results, key=lambda x: x['overall_score'])
            print(f"\n🏆 Best Overall: {best_overall['index_name']}")
            print(f"   Search: {best_overall['avg_search_time_ms']:.2f}ms")
            print(f"   Creation: {best_overall['creation_time']:.3f}s") 
            print(f"   Memory: {best_overall['memory_usage_mb']:.1f}MB")

def main():
    benchmark = VectorBenchmark()
    benchmark.run_comprehensive_benchmark()

if __name__ == "__main__":
    main() 