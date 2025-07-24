#!/usr/bin/env python3
"""
Test script to verify location functionality works when manually provided
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_app import chat_response, initialize_agent, generate_session_id
import uuid

def test_manual_location_workflow():
    """Test the manual location workflow"""
    print("🧪 Testing Manual Location Workflow")
    print("=" * 60)
    
    # Initialize agent
    print("🔄 Initializing agent...")
    init_result = initialize_agent()
    if "Error" in init_result:
        print(f"❌ {init_result}")
        return False
    
    print("✅ Agent initialized successfully")
    
    # Generate session ID
    session_id = generate_session_id()
    print(f"📝 Session ID: {session_id}")
    
    # Test cases
    test_cases = [
        {
            "name": "Location query with coordinates",
            "location": "37.6922,-97.3375 (Wichita, KS)",
            "message": "Where is the nearest DMV office to me?"
        },
        {
            "name": "Location query with city name",
            "location": "Lawrence, KS",
            "message": "Where is the nearest DMV office to me?"
        },
        {
            "name": "Location query without location",
            "location": "Location not available",
            "message": "Where is the nearest DMV office to me?"
        },
        {
            "name": "General DMV question with location",
            "location": "39.0119,-95.7222 (Topeka, KS)",
            "message": "How do I renew my driver's license?"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        print(f"📍 Location: {test_case['location']}")
        print(f"💬 Message: {test_case['message']}")
        print("\n🤖 Response:")
        
        try:
            # Use the chat_response function directly
            _, history = chat_response(
                message=test_case['message'],
                history=[],
                language="en",
                session_id=f"{session_id}_{i}",
                location=test_case['location']
            )
            
            # Get the response from history
            if history:
                response = history[-1][1]
                print(response)
            else:
                print("❌ No response generated")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "=" * 60)
    
    print("✅ Manual location workflow test completed")
    return True

if __name__ == "__main__":
    success = test_manual_location_workflow()
    sys.exit(0 if success else 1) 