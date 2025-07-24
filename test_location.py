#!/usr/bin/env python3
"""
Test script to verify location functionality
"""
import os
import sys
from agent import RAGAgent

def test_location_functionality():
    """Test the location tool directly"""
    print("🧪 Testing Location Functionality")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("🔄 Initializing RAG Agent...")
        agent = RAGAgent()
        print("✅ Agent initialized successfully")
        
        # Test cases
        test_cases = [
            {
                "name": "Direct location query with coordinates",
                "message": "User's current location: 37.6922,-97.3375\n\nUser's question: Where is the nearest DMV office to me?"
            },
            {
                "name": "Direct location query with city",
                "message": "User's current location: Wichita, KS\n\nUser's question: Where is the nearest DMV office to me?"
            },
            {
                "name": "Simple location query",
                "message": "Where is the nearest DMV office to me?"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['name']}")
            print("-" * 50)
            print(f"📝 Input: {test_case['message']}")
            print("\n🤖 Agent Response:")
            
            try:
                response = agent.chat(test_case['message'], f"test_session_{i}")
                print(response)
                print("\n" + "=" * 50)
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("=" * 50)
        
        print("\n🎯 Testing location tool directly...")
        print("-" * 50)
        
        # Test the location tool directly
        try:
            location_tool = agent.location_tool
            result = location_tool.func("37.6922,-97.3375")
            print("✅ Direct location tool test:")
            print(result)
        except Exception as e:
            print(f"❌ Direct tool test failed: {str(e)}")
            
    except Exception as e:
        print(f"❌ Failed to initialize or test: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_location_functionality()
    sys.exit(0 if success else 1) 