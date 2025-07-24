#!/usr/bin/env python3
"""
Debug script to identify location issues
"""
import gradio as gr
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import RAGAgent

# Global agent
agent = None

def initialize_agent():
    global agent
    try:
        if agent is None:
            agent = RAGAgent()
        return "✅ Agent ready!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def test_location_query(location, message):
    """Test function to verify location processing"""
    global agent
    
    print(f"🔍 DEBUG: Location received: '{location}'")
    print(f"🔍 DEBUG: Message received: '{message}'")
    
    if agent is None:
        init_result = initialize_agent()
        if "Error" in init_result:
            return f"❌ {init_result}"
    
    try:
        # Prepare enhanced message like in gradio_app.py
        enhanced_message = message
        if location and location != "Location not available":
            # Check if user is asking about DMV office locations
            location_keywords = ["dmv office", "nearest dmv", "dmv location", "where is", "office", "nearest office"]
            is_location_query = any(keyword in message.lower() for keyword in location_keywords)
            
            if is_location_query:
                enhanced_message = f"User's current location: {location}\n\nUser's question: {message}\n\nNote: The user is asking about DMV office locations. Use the find_nearest_dmv tool with the provided location."
            else:
                enhanced_message = f"User's current location: {location}\n\nUser's question: {message}"
        
        print(f"🔍 DEBUG: Enhanced message: '{enhanced_message}'")
        
        # Get response from agent
        response = agent.chat(enhanced_message, "debug_session")
        
        print(f"🔍 DEBUG: Agent response: '{response[:200]}...'")
        
        return response
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"❌ DEBUG: {error_msg}")
        return error_msg

def create_debug_interface():
    """Create a simple debug interface"""
    
    with gr.Blocks(title="Location Debug") as app:
        gr.Markdown("# Location Debug Interface")
        
        # Initialize agent
        init_result = initialize_agent()
        gr.Markdown(f"**Agent Status:** {init_result}")
        
        with gr.Row():
            location_input = gr.Textbox(
                label="Location (manually enter for testing)",
                placeholder="Enter location like 'Wichita, KS' or '37.6922,-97.3375'",
                value="Location not available"
            )
            
        message_input = gr.Textbox(
            label="Message",
            placeholder="Enter your question",
            value="Where is the nearest DMV office to me?"
        )
        
        submit_btn = gr.Button("🧪 Test Location Query")
        
        output = gr.Textbox(
            label="Debug Output",
            lines=20,
            max_lines=30
        )
        
        # Test button
        submit_btn.click(
            test_location_query,
            inputs=[location_input, message_input],
            outputs=[output]
        )
        
        # Pre-populated test cases
        gr.Markdown("## Quick Test Cases")
        
        test_cases = [
            ("37.6922,-97.3375 (Wichita, KS)", "Where is the nearest DMV office to me?"),
            ("Lawrence, KS", "Where is the nearest DMV office to me?"),
            ("Location not available", "Where is the nearest DMV office to me?"),
            ("Topeka, KS", "How do I renew my driver's license?")
        ]
        
        for i, (test_location, test_message) in enumerate(test_cases, 1):
            test_btn = gr.Button(f"Test {i}: {test_location[:20]}...")
            test_btn.click(
                test_location_query,
                inputs=[gr.State(test_location), gr.State(test_message)],
                outputs=[output]
            )
    
    return app

if __name__ == "__main__":
    app = create_debug_interface()
    app.launch(server_name="localhost", server_port=7862, share=False) 