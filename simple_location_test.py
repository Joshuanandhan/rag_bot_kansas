#!/usr/bin/env python3
"""
Simple location test for users to verify the location functionality works
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

def test_location_with_agent(location, message):
    """Test location functionality directly"""
    global agent
    
    if agent is None:
        init_result = initialize_agent()
        if "Error" in init_result:
            return f"❌ {init_result}"
    
    try:
        # Prepare enhanced message exactly like in gradio_app.py
        enhanced_message = message
        if location and location.strip() and location != "Location not available":
            # Check if user is asking about DMV office locations
            location_keywords = ["dmv office", "nearest dmv", "dmv location", "where is", "office", "nearest office"]
            is_location_query = any(keyword in message.lower() for keyword in location_keywords)
            
            if is_location_query:
                enhanced_message = f"User's current location: {location}\n\nUser's question: {message}\n\nNote: The user is asking about DMV office locations. Use the find_nearest_dmv tool with the provided location."
            else:
                enhanced_message = f"User's current location: {location}\n\nUser's question: {message}"
        
        # Get response from agent
        response = agent.chat(enhanced_message, "test_session")
        
        return response
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def create_simple_location_test():
    """Create a simple location test interface"""
    
    with gr.Blocks(title="Kansas DMV Location Test") as app:
        gr.Markdown("# Kansas DMV Location Test")
        gr.Markdown("**Test the location functionality directly by entering your location below.**")
        
        # Initialize agent
        init_result = initialize_agent()
        gr.Markdown(f"**Agent Status:** {init_result}")
        
        with gr.Row():
            location_input = gr.Textbox(
                label="📍 Your Location",
                placeholder="Enter your location: 'Wichita, KS' or '37.6922,-97.3375'",
                value="",
                scale=2
            )
            message_input = gr.Textbox(
                label="💬 Your Question",
                placeholder="Ask about DMV offices...",
                value="Where is the nearest DMV office to me?",
                scale=2
            )
        
        submit_btn = gr.Button("🔍 Find Nearest DMV Office", variant="primary")
        
        output = gr.Textbox(
            label="🤖 Assistant Response",
            lines=15,
            max_lines=25
        )
        
        # Submit button
        submit_btn.click(
            test_location_with_agent,
            inputs=[location_input, message_input],
            outputs=[output]
        )
        
        # Pre-filled test buttons
        gr.Markdown("## 🧪 Quick Test Cases")
        
        with gr.Row():
            test1_btn = gr.Button("Test Wichita")
            test2_btn = gr.Button("Test Lawrence")
            test3_btn = gr.Button("Test Topeka")
            test4_btn = gr.Button("Test No Location")
        
        # Test button functions
        test1_btn.click(
            lambda: test_location_with_agent("37.6922,-97.3375 (Wichita, KS)", "Where is the nearest DMV office to me?"),
            outputs=[output]
        )
        
        test2_btn.click(
            lambda: test_location_with_agent("Lawrence, KS", "Where is the nearest DMV office to me?"),
            outputs=[output]
        )
        
        test3_btn.click(
            lambda: test_location_with_agent("Topeka, KS", "How do I renew my driver's license?"),
            outputs=[output]
        )
        
        test4_btn.click(
            lambda: test_location_with_agent("Location not available", "Where is the nearest DMV office to me?"),
            outputs=[output]
        )
        
        # Instructions
        gr.Markdown("""
        ### 📋 **How to Test:**
        
        1. **Enter your location** in the location field (e.g., "Wichita, KS" or "37.6922,-97.3375")
        2. **Ask about DMV offices** (the default question should work)
        3. **Click "Find Nearest DMV Office"**
        4. You should see the nearest DMV offices with addresses, phone numbers, and distances
        
        ### 🎯 **Expected Results:**
        - **With location**: List of nearest DMV offices with distances
        - **Without location**: Request to provide location information
        
        ### 🔧 **Sample Locations to Test:**
        - `Wichita, KS`
        - `Lawrence, KS`
        - `Topeka, KS`
        - `37.6922,-97.3375`
        - `39.0119,-95.7222`
        """)
    
    return app

if __name__ == "__main__":
    app = create_simple_location_test()
    app.launch(server_name="localhost", server_port=7864, share=False) 