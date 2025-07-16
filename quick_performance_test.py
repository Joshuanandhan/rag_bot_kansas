"""
Quick Performance Test
A simplified version to quickly test key configurations
"""

import time
import glob
import os
from typing import Dict, List
from datetime import datetime
import pandas as pd

# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    
    def cleanup_old_results(self):
        """Remove old quick test result files"""
        print("🧹 Cleaning up old quick test files...")
        
        # Patterns to match result files
        patterns = [
            "quick_test_results_*.json",
            "quick_test_*_dashboard.png",
            "quick_test_*_comparison.png",
            "quick_test_*_terminal_summary.txt"
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
    
    def generate_quick_test_graphs(self, results: List[Dict], filename: str):
        """Generate graphs for quick test results"""
        
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            print("⚠️  No successful results to graph")
            return []
        
        df = pd.DataFrame(successful_results)
        
        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quick Performance Test Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Total Time Comparison
        ax1 = axes[0, 0]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax1.bar(df['config_name'], df['total_time'], color=colors, alpha=0.8)
        ax1.set_title('Total Response Time by Configuration')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Configuration')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 2. Time Breakdown
        ax2 = axes[0, 1]
        time_components = ['setup_time', 'retrieval_time', 'generation_time']
        bottom = [0] * len(df)
        colors_breakdown = ['#FFE66D', '#FF6B6B', '#4ECDC4']
        
        for i, component in enumerate(time_components):
            ax2.bar(df['config_name'], df[component], bottom=bottom, 
                   label=component.replace('_', ' ').title(), color=colors_breakdown[i], alpha=0.8)
            bottom = [b + df[component].iloc[j] for j, b in enumerate(bottom)]
        
        ax2.set_title('Time Breakdown by Component')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xlabel('Configuration')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Cost Comparison
        ax3 = axes[1, 0]
        costs = []
        for _, row in df.iterrows():
            config = self.get_config_by_name(row['config_name'])
            cost = self.estimate_cost(row.to_dict(), config)
            costs.append(cost)
        
        bars = ax3.bar(df['config_name'], costs, color=['#FF9F43', '#6C5CE7', '#A29BFE'], alpha=0.8)
        ax3.set_title('Estimated Cost per 1000 Queries')
        ax3.set_ylabel('Cost ($)')
        ax3.set_xlabel('Configuration')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'${cost:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance vs Cost Scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['total_time'], costs, 
                            c=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
                            s=100, alpha=0.8)
        
        # Add labels for each point
        for i, (_, row) in enumerate(df.iterrows()):
            ax4.annotate(row['config_name'], 
                        (row['total_time'], costs[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left')
        
        ax4.set_title('Performance vs Cost Trade-off')
        ax4.set_xlabel('Response Time (seconds)')
        ax4.set_ylabel('Cost per 1000 queries ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = f"{filename}_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Quick test dashboard: {dashboard_file}")
        return [dashboard_file]
    
    def save_terminal_summary(self, results: List[Dict], filename: str):
        """Save terminal summary for quick test"""
        
        summary_file = f"{filename}_terminal_summary.txt"
        successful_results = [r for r in results if r.get("success", False)]
        
        with open(summary_file, 'w') as f:
            f.write("🎯 QUICK PERFORMANCE TEST TERMINAL SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            if successful_results:
                # Sort by total time
                successful_results.sort(key=lambda x: x["total_time"])
                
                f.write("🏆 PERFORMANCE RANKING:\n")
                for i, result in enumerate(successful_results, 1):
                    f.write(f"{i}. {result['config_name']}: {result['total_time']:.2f}s\n")
                
                fastest = successful_results[0]
                f.write(f"\n🥇 FASTEST CONFIGURATION: {fastest['config_name']} ({fastest['total_time']:.2f}s)\n")
                f.write(f"   ⚡ Retrieval: {fastest['retrieval_time']:.2f}s\n")
                f.write(f"   🤖 Generation: {fastest['generation_time']:.2f}s\n")
                f.write(f"   📝 Response length: {fastest['response_length']} chars\n\n")
                
                # Cost estimates
                f.write("💰 ESTIMATED COSTS PER 1000 QUERIES:\n")
                for result in successful_results:
                    config = self.get_config_by_name(result['config_name'])
                    cost = self.estimate_cost(result, config)
                    f.write(f"   {result['config_name']}: ~${cost:.2f}\n")
            
            f.write(f"\n📁 GENERATED FILES:\n")
            f.write(f"   📄 Raw results: {filename}.json\n")
            f.write(f"   📊 Dashboard: {filename}_dashboard.png\n")
            f.write(f"   📋 Terminal summary: {filename}_terminal_summary.txt\n")
            
            f.write(f"\n🔍 NEXT STEPS:\n")
            f.write("1. View the dashboard PNG file for visual analysis\n")
            f.write("2. Use the fastest configuration for your use case\n")
            f.write("3. Consider running 'comprehensive' test for detailed analysis\n")
            f.write("4. Monitor actual costs in your OpenAI usage dashboard\n")
        
        return summary_file

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
        
        # Clean up old results
        self.cleanup_old_results()
        print()
        
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
        
        # Save results and generate graphs
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"quick_test_results_{timestamp}.json"
        filename_base = f"quick_test_{timestamp}"
        
        # Save JSON results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate graphs
        print("\n📊 Generating performance visualizations...")
        try:
            graph_files = self.generate_quick_test_graphs(results, filename_base)
            terminal_summary = self.save_terminal_summary(results, filename_base)
            
            print(f"\n✅ Quick test completed! Files saved:")
            print(f"   📄 Raw results: {results_file}")
            print(f"   📊 Dashboard: {filename_base}_dashboard.png")
            print(f"   📋 Terminal summary: {terminal_summary}")
            print(f"\n💡 TIP: Open the PNG file to view the performance dashboard!")
            
        except Exception as e:
            print(f"⚠️  Error generating graphs: {e}")
            print(f"   📄 Raw results saved to: {results_file}")
            print("   Continuing without graphs...")
    
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