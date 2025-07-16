#!/usr/bin/env python3
"""
Performance Test Runner
Easy way to run different performance tests on your RAG system
"""

import sys
import subprocess
import argparse
from pathlib import Path

class TestRunner:
    def __init__(self):
        self.tests = {
            "quick": {
                "script": "quick_performance_test.py",
                "description": "Quick test of 3 key configurations (Fast & Cheap, Balanced, High Quality)",
                "time": "~2-3 minutes",
                "cost": "Low (~$0.10)",
                "features": "Basic performance comparison + dashboard graphs"
            },
            "vector": {
                "script": "vector_benchmark.py", 
                "description": "Benchmark different vector search algorithms (FAISS indexes)",
                "time": "~5-10 minutes",
                "cost": "Medium (~$0.20)",
                "features": "Vector search optimization + performance graphs"
            },
            "comprehensive": {
                "script": "performance_tester.py",
                "description": "Full performance test across all model combinations",
                "time": "~30-60 minutes",
                "cost": "High (~$2-5)",
                "features": "Complete analysis + performance graphs + cost analysis"
            }
        }
    
    def display_menu(self):
        """Display the test menu"""
        print("🔬 RAG Performance Test Runner")
        print("=" * 50)
        print()
        
        for key, test in self.tests.items():
            print(f"📊 {key.upper()} TEST")
            print(f"   Description: {test['description']}")
            print(f"   Time: {test['time']}")
            print(f"   Cost: {test['cost']}")
            print(f"   Features: {test['features']}")
            print()
    
    def check_prerequisites(self):
        """Check if prerequisites are met"""
        issues = []
        
        # Check if output_all directory exists
        if not Path("output_all").exists():
            issues.append("❌ No 'output_all' directory found. Run PDF processing or web scraping first.")
        else:
            # Check if there are any processed documents
            has_docs = False
            for source_dir in Path("output_all").iterdir():
                if source_dir.is_dir() and (source_dir / "chunks").exists():
                    chunk_files = list((source_dir / "chunks").glob("*.md"))
                    if chunk_files:
                        has_docs = True
                        break
            
            if not has_docs:
                issues.append("❌ No processed documents found in output_all. Process some content first.")
        
        # Check for .env file
        if not Path(".env").exists():
            issues.append("❌ No .env file found. Make sure you have OPENAI_API_KEY set.")
        
        # Check Python packages
        required_packages = {
            'openai': 'openai',
            'langchain': 'langchain',
            'faiss': 'faiss-cpu',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn'
        }
        
        for package, install_name in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                issues.append(f"❌ Missing Python package: {install_name}. Run 'pip install -r requirements.txt'")
        
        if issues:
            print("🚨 PREREQUISITES CHECK FAILED:")
            print()
            for issue in issues:
                print(f"   {issue}")
            print()
            print("Please fix these issues before running performance tests.")
            return False
        
        print("✅ Prerequisites check passed!")
        return True
    
    def run_test(self, test_type: str, args: list = None):
        """Run a specific test"""
        
        if test_type not in self.tests:
            print(f"❌ Unknown test type: {test_type}")
            print(f"Available tests: {', '.join(self.tests.keys())}")
            return False
        
        test_info = self.tests[test_type]
        script = test_info["script"]
        
        if not Path(script).exists():
            print(f"❌ Test script not found: {script}")
            return False
        
        print(f"🚀 Running {test_type.upper()} performance test...")
        print(f"📄 Script: {script}")
        print(f"⏱️  Expected time: {test_info['time']}")
        print(f"💰 Expected cost: {test_info['cost']}")
        print(f"🎯 Features: {test_info['features']}")
        print()
        
        # Build command
        cmd = [sys.executable, script]
        if args:
            cmd.extend(args)
        
        try:
            # Run the test
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"\n✅ {test_type.upper()} test completed successfully!")
            
            # Provide next steps based on test type
            if test_type == "comprehensive":
                print("\n📊 Next steps:")
                print("1. Review the generated performance dashboard (PNG files)")
                print("2. Read the detailed summary report (MD file)")
                print("3. Check the CSV file for raw data analysis")
                print("4. Use the fastest configuration in your production system")
            elif test_type == "quick":
                print("\n📊 Next steps:")
                print("1. Review the generated performance dashboard (PNG file)")
                print("2. Read the terminal summary (TXT file)")
                print("3. Use the fastest configuration for your use case")
                print("4. Consider running 'comprehensive' test for detailed analysis")
            elif test_type == "vector":
                print("\n📊 Next steps:")
                print("1. Review the generated performance dashboard (PNG file)")
                print("2. Read the terminal summary (TXT file)")
                print("3. Use the best overall index type for your use case")
                print("4. Consider the trade-offs between speed, memory, and creation time")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {test_type.upper()} test failed with exit code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print(f"\n🛑 {test_type.upper()} test interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Error running {test_type.upper()} test: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="RAG Performance Test Runner")
    
    parser.add_argument("test_type", nargs='?', 
                       choices=["quick", "vector", "comprehensive", "menu"],
                       default="menu",
                       help="Type of test to run (default: show menu)")
    
    # Options for comprehensive test
    parser.add_argument("--sample-size", type=int, default=3,
                       help="Number of queries per configuration (comprehensive test only)")
    parser.add_argument("--output", type=str, 
                       help="Output filename prefix")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Skip cleanup of old result files (comprehensive test only)")
    
    args, unknown = parser.parse_known_args()
    
    runner = TestRunner()
    
    # Show menu if requested or no test specified
    if args.test_type == "menu":
        runner.display_menu()
        
        print("Usage:")
        print("  python run_performance_tests.py quick       # Quick test")
        print("  python run_performance_tests.py vector      # Vector benchmark")
        print("  python run_performance_tests.py comprehensive  # Full test with graphs")
        print()
        print("Examples:")
        print("  python run_performance_tests.py comprehensive --sample-size 5")
        print("  python run_performance_tests.py comprehensive --no-cleanup")
        print("  python run_performance_tests.py quick")
        
        print("\n🆕 New Features:")
        print("  • All tests now generate performance graphs and visualizations")
        print("  • Automatic cleanup of old result files (use --no-cleanup to disable)")
        print("  • Enhanced summary reports with insights and recommendations")
        print("  • Terminal summaries saved to text files for all tests")
        print("  • Performance dashboards for quick visual analysis")
        return
    
    # Check prerequisites
    if not runner.check_prerequisites():
        sys.exit(1)
    
    # Prepare test-specific arguments
    test_args = []
    
    if args.test_type == "comprehensive":
        if args.sample_size:
            test_args.extend(["--sample-size", str(args.sample_size)])
        if args.output:
            test_args.extend(["--output", args.output])
        if args.no_cleanup:
            test_args.append("--no-cleanup")
    
    # Add any unknown arguments
    test_args.extend(unknown)
    
    # Confirm before running expensive tests
    if args.test_type == "comprehensive":
        print("⚠️  COMPREHENSIVE TEST WARNING:")
        print(f"   This will test many configurations and may take 30-60 minutes")
        print(f"   Estimated cost: $2-5 in OpenAI API usage")
        print(f"   Sample size: {args.sample_size} queries per configuration")
        print(f"   Will generate: JSON, CSV, MD summary, and PNG graph files")
        
        if not args.no_cleanup:
            print(f"   Will clean up old result files automatically")
        
        response = input("\nContinue? (y/N): ").strip().lower()
        if response != 'y':
            print("Test cancelled.")
            return
    
    # Run the test
    success = runner.run_test(args.test_type, test_args)
    
    if success:
        print("\n🎉 Performance testing completed!")
        print("\nNext steps:")
        print("1. Review the generated reports and JSON files")
        
        if args.test_type == "comprehensive":
            print("2. Check the performance dashboard graphs (PNG files)")
            print("3. Read the detailed summary report for insights")
            print("4. Use the fastest configuration for your production agent")
            print("5. Consider cost vs performance trade-offs from the analysis")
        elif args.test_type == "quick":
            print("2. Check the performance dashboard graph (PNG file)")
            print("3. Read the terminal summary for quick insights")
            print("4. Use the fastest configuration for your production agent")
            print("5. Run 'comprehensive' test for detailed analysis with more graphs")
        elif args.test_type == "vector":
            print("2. Check the performance dashboard graph (PNG file)")
            print("3. Read the terminal summary for algorithm insights")
            print("4. Use the best overall index type for your production agent")
            print("5. Run 'comprehensive' test for full model analysis")
        else:
            print("2. Use the fastest configuration for your production agent")
            print("3. Consider cost vs performance trade-offs")
        
        print("\n💡 All tests now generate performance graphs saved as PNG files!")
    else:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure your .env file has OPENAI_API_KEY")
        print("2. Check that you have processed documents in output_all/")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")
        print("4. Ensure matplotlib and seaborn are installed for graph generation")
        sys.exit(1)

if __name__ == "__main__":
    main() 