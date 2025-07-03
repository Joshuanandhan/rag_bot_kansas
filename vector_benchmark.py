"""
Vector Search Benchmark
Tests different vector search algorithms and indexing strategies
"""

import time
import numpy as np
from typing import List, Dict, Any
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from pathlib import Path
import json

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
        
        # Create embeddings
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")  # Use small for speed
        
        print("🔄 Creating embeddings...")
        texts = [doc.page_content for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        print(f"✅ Created {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        return documents, np.array(embeddings)
    
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
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"vector_benchmark_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {results_file}")
    
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