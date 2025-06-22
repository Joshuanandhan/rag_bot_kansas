"""
Simple Gradio Web App for Kansas Driving Knowledge RAG Agent
"""
import gradio as gr
from agent import RAGAgent
from typing import List, Tuple

# Global agent instance
agent = None

def initialize_agent():
    """Initialize the RAG agent."""
    global agent
    try:
        if agent is None:
            agent = RAGAgent()
        return "✅ Agent ready!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def chat_response(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Process chat message and return response."""
    global agent
    
    if not message.strip():
        return "", history
    
    # Initialize agent if needed
    if agent is None:
        init_result = initialize_agent()
        if "Error" in init_result:
            history.append((message, init_result))
            return "", history
    
    try:
        # Get response from agent
        response = agent.chat(message, "gradio_session")
        history.append((message, response))
        return "", history
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        return "", history

def create_app():
    """Create the Gradio interface."""
    
    # Example questions
    examples = [
        "What are the requirements to get a commercial driver's license in Kansas?",
        "What is the minimum age for a motorcycle license?",
        "What are the speed limits on Kansas highways?",
        "What documents do I need for a driving test?",
        "What are the penalties for DUI in Kansas?"
    ]
    
    with gr.Blocks(title="Kansas Driving Assistant") as app:
        
        gr.HTML("""
            <h1 style="text-align: center; color: #1976d2;">
                🚗 Kansas Driving Knowledge Assistant
            </h1>
            <p style="text-align: center; color: #666;">
                Ask me anything about Kansas driving regulations, CDL requirements, or motorcycle laws!
            </p>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=400,
                    show_copy_button=True,
                    placeholder="Start by asking a question about Kansas driving..."
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question here...",
                        container=False,
                        scale=4
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)
                
                clear = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                gr.HTML("<h3>📝 Quick Examples</h3>")
                
                for example in examples:
                    gr.Button(
                        example,
                        size="sm"
                    ).click(
                        lambda x=example: x,
                        outputs=msg
                    )
        
        gr.HTML("""
            <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                <h4>About this Assistant</h4>
                <p>This AI assistant uses advanced retrieval technology to provide accurate information from Kansas driving handbooks including:</p>
                <ul>
                    <li>📖 Kansas Driving Handbook</li>
                    <li>🚛 Commercial Driver's License Manual</li>
                    <li>🏍️ Motorcycle Handbook</li>
                </ul>
            </div>
        """)
        
        # Event handlers
        msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
        submit.click(chat_response, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], outputs=chatbot)
    
    return app

if __name__ == "__main__":
    print("\n🚀 Starting Kansas Driving Knowledge Assistant...")
    print("📍 Initializing RAG Agent...")
    
    # Pre-initialize the agent
    init_status = initialize_agent()
    print(f"   {init_status}")
    
    print("\n🌐 Launching web interface...")
    print("   Access at: http://localhost:7860")
    print("   Press Ctrl+C to stop\n")
    
    app = create_app()
    
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
