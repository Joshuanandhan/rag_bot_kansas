"""
Quick Performance Test
A simplified version to quickly test key configurations
"""

import time
from typing import Dict, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from pathlib import Path
import json

class QuickTester:
    def __init__(self):
        self.test_queries = [
            "What are the requirements for a commercial driver's license?",
            "How do I renew my driver's license?",
            "What documents do I need for vehicle registration?"
        ]
        
        # Quick test configurations
        self.test_configs = [
            {
                "name": "Fast & Cheap",
                "embedding": "text-embedding-3-small",
                "llm": "gpt-3.5-turbo",
                "chunks": 3
            },
            {
                "name": "Balanced",
                "embedding": "text-embedding-3-large", 
                "llm": "gpt-4o-mini",
                "chunks": 5
            },
            {
                "name": "High Quality",
                "embedding": "text-embedding-3-large",
                "llm": "gpt-4o",
                "chunks": 7
            }
        ]
    
    def load_documents(self):
        """Load documents from output_all directory"""
        documents = []
        output_path = Path("output_all")
        
        if not output_path.exists():
            print("❌ No output_all directory found. Please process some data first.")
            return []
        
        for source_dir in output_path.iterdir():
            if source_dir.is_dir():
                chunks_dir = source_dir / "chunks"
                if chunks_dir.exists():
                    loader = DirectoryLoader(str(chunks_dir), glob="**/*.md")
                    docs = loader.load()
                    documents.extend(docs)
        
        print(f"📄 Loaded {len(documents)} documents")
        return documents
    
    def test_configuration(self, config: Dict, documents: List, query: str) -> Dict:
        """Test a single configuration"""
        print(f"  🔧 Testing {config['name']} configuration...")
        
        start_time = time.time()
        
        try:
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(model=config["embedding"])
            vector_store = FAISS.from_documents(documents, embeddings)
            
            setup_time = time.time() - start_time
            
            # Test retrieval
            retrieval_start = time.time()
            results = vector_store.similarity_search(query, k=config["chunks"])
            retrieval_time = time.time() - retrieval_start
            
            # Test generation
            context = "\n\n".join([doc.page_content[:400] for doc in results])
            
            generation_start = time.time()
            llm = ChatOpenAI(model=config["llm"], temperature=0.7)
            
            prompt = f"""Based on the context below, answer the question:

Context:
{context}

Question: {query}

Answer:"""
            
            response = llm.invoke(prompt)
            generation_time = time.time() - generation_start
            
            total_time = setup_time + retrieval_time + generation_time
            
            return {
                "config_name": config["name"],
                "setup_time": setup_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "chunks_found": len(results),
                "response_length": len(response.content),
                "success": True,
                "response_preview": response.content[:150] + "..." if len(response.content) > 150 else response.content
            }
            
        except Exception as e:
            return {
                "config_name": config["name"],
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    def run_quick_test(self):
        """Run quick performance test"""
        print("🚀 Quick Performance Test")
        print("=" * 40)
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            return
        
        # Use a subset of documents for quick testing
        test_docs = documents[:50] if len(documents) > 50 else documents
        print(f"📊 Using {len(test_docs)} documents for testing")
        
        results = []
        
        # Test each configuration with first query
        test_query = self.test_queries[0]
        print(f"\n🔍 Test Query: '{test_query}'")
        print("-" * 40)
        
        for config in self.test_configs:
            result = self.test_configuration(config, test_docs, test_query)
            results.append(result)
            
            if result["success"]:
                print(f"  ✅ {result['config_name']}: {result['total_time']:.2f}s total")
                print(f"     ⚡ Retrieval: {result['retrieval_time']:.2f}s | Generation: {result['generation_time']:.2f}s")
                print(f"     📝 Response: {result['response_preview']}")
            else:
                print(f"  ❌ {result['config_name']}: Failed - {result['error']}")
            print()
        
        # Show summary
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            print("🏆 SUMMARY:")
            print("-" * 40)
            
            # Sort by total time
            successful_results.sort(key=lambda x: x["total_time"])
            
            for i, result in enumerate(successful_results, 1):
                print(f"{i}. {result['config_name']}: {result['total_time']:.2f}s")
            
            fastest = successful_results[0]
            print(f"\n🥇 Fastest: {fastest['config_name']} ({fastest['total_time']:.2f}s)")
            
            # Cost estimate
            print(f"\n💰 Estimated costs per 1000 queries:")
            for result in successful_results:
                cost = self.estimate_cost(result, self.get_config_by_name(result['config_name']))
                print(f"   {result['config_name']}: ~${cost:.2f}")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"quick_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Results saved to: {results_file}")
    
    def get_config_by_name(self, name: str) -> Dict:
        """Get configuration by name"""
        for config in self.test_configs:
            if config["name"] == name:
                return config
        return {}
    
    def estimate_cost(self, result: Dict, config: Dict) -> float:
        """Estimate cost per 1000 queries"""
        # Rough cost estimates (as of 2024)
        embedding_costs = {
            "text-embedding-3-small": 0.02,  # per 1M tokens
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10
        }
        
        llm_costs = {
            "gpt-3.5-turbo": 0.50,  # per 1M tokens (input)
            "gpt-4o-mini": 0.15,
            "gpt-4o": 5.00
        }
        
        # Estimate tokens (very rough)
        embedding_tokens = 500  # Average query + context
        llm_input_tokens = 1500  # Context + query
        
        embedding_cost = (embedding_tokens / 1_000_000) * embedding_costs.get(config.get("embedding", ""), 0.02)
        llm_cost = (llm_input_tokens / 1_000_000) * llm_costs.get(config.get("llm", ""), 0.50)
        
        return (embedding_cost + llm_cost) * 1000

def main():
    tester = QuickTester()
    tester.run_quick_test()

if __name__ == "__main__":
    main() 