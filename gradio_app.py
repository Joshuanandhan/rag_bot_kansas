"""
Simple Gradio Web App for Kansas Driving Knowledge RAG Agent
"""
import gradio as gr
import os
from dotenv import load_dotenv
from agent import RAGAgent
from typing import List, Tuple
from deep_translator import GoogleTranslator

# Load environment variables from .env file
load_dotenv()

def translate_text(text, target_lang):
    """Translate text using Google Translator."""
    try:
        if target_lang == "en":
            return text
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        return text  # Return original if translation fails

# Global agent instance
agent = None

def initialize_agent():
    """Initialize the RAG agent."""
    global agent
    try:
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            return "❌ Error: OPENAI_API_KEY not found. Please add it to your .env file."
        
        if agent is None:
            agent = RAGAgent()
        return "✅ Agent ready!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def chat_response(message: str, history: List[Tuple[str, str]], language: str = "en") -> Tuple[str, List[Tuple[str, str]]]:
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
        
        # Translate response if Spanish is selected
        if language == "es":
            response = translate_text(response, "es")
        
        history.append((message, response))
        return "", history
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        return "", history

def create_app():
    """Create the modern ChatGPT-inspired interface."""
    
    # Custom CSS matching the screenshot design
    custom_css = """
    /* Global Styles */
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh !important;
        margin: 0 !important;
        padding: 20px !important;
    }
    
    /* Main Container */
    .main-container {
        max-width: 900px !important;
        width: 90% !important;
        height: 90vh !important;
        margin: 0 auto !important;
        background: white !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Header Styles */
    .dmv-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        padding: 20px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    
    .dmv-header .gradio-row {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    .header-left {
        display: flex !important;
        align-items: center !important;
        gap: 15px !important;
    }
    
    .ks-logo {
        width: 50px !important;
        height: 50px !important;
        background: white !important;
        border-radius: 12px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: bold !important;
        color: #2a5298 !important;
        font-size: 18px !important;
    }
    
    .title-section h1 {
        font-size: 24px !important;
        font-weight: bold !important;
        margin: 0 0 4px 0 !important;
        color: white !important;
    }
    
    .title-section p {
        font-size: 14px !important;
        opacity: 0.9 !important;
        margin: 0 !important;
        color: white !important;
    }
    
    .header-controls {
        display: flex !important;
        gap: 10px !important;
        align-items: center !important;
    }
    
    .glass-btn {
        background: rgba(255,255,255,0.2) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    .glass-btn:hover {
        background: rgba(255,255,255,0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Chat Area */
    .chat-container {
        flex: 1 !important;
        background: #f8f9fa !important;
        overflow-y: auto !important;
        padding: 20px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Welcome Screen */
    .welcome-screen {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        padding: 20px !important;
        height: 100% !important;
    }
    
    .welcome-icon {
        width: 80px !important;
        height: 80px !important;
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 40px !important;
        margin-bottom: 20px !important;
        animation: pulse 2s infinite !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .welcome-title {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #2c3e50 !important;
        margin-bottom: 10px !important;
        margin-top: 0 !important;
    }
    
    .welcome-subtitle {
        font-size: 16px !important;
        color: #7f8c8d !important;
        margin-bottom: 30px !important;
        line-height: 1.5 !important;
    }
    
    /* Sample Prompts Grid */
    .prompts-grid {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)) !important;
        gap: 15px !important;
        width: 100% !important;
        max-width: 600px !important;
        margin-top: 20px !important;
    }
    
    .prompt-card {
        background: white !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border: 2px solid #e9ecef !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        text-align: left !important;
    }
    
    .prompt-card:hover {
        border-color: #667eea !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    }
    
    .prompt-card h4 {
        font-size: 16px !important;
        font-weight: bold !important;
        color: #2c3e50 !important;
        margin: 0 0 8px 0 !important;
    }
    
    .prompt-card p {
        font-size: 14px !important;
        color: #7f8c8d !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }
    
    /* Chatbot Styles */
    .chatbot {
        border: none !important;
        background: transparent !important;
        flex: 1 !important;
    }
    
    /* Input Area */
    .input-container {
        background: white !important;
        border-top: 1px solid #e9ecef !important;
        padding: 20px !important;
        display: flex !important;
        gap: 10px !important;
        align-items: flex-end !important;
    }
    
    .input-box {
        flex: 1 !important;
        border: 2px solid #e9ecef !important;
        border-radius: 25px !important;
        padding: 15px 20px !important;
        font-size: 16px !important;
        resize: none !important;
        transition: all 0.3s ease !important;
        min-height: 50px !important;
        max-height: 100px !important;
    }
    
    .input-box:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    .send-button {
        width: 50px !important;
        height: 50px !important;
        border-radius: 50% !important;
        background: #667eea !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 18px !important;
        transition: all 0.3s ease !important;
    }
    
    .send-button:hover {
        background: #5a67d8 !important;
        transform: scale(1.1) !important;
    }
    
    /* Hide default Gradio elements */
    .gradio-container .contain, .gradio-container .panel {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-container {
            width: 95% !important;
            height: 95vh !important;
        }
        
        .dmv-header {
            padding: 15px !important;
        }
        
        .header-left {
            gap: 10px !important;
        }
        
        .title-section h1 {
            font-size: 20px !important;
        }
        
        .prompts-grid {
            grid-template-columns: 1fr !important;
        }
        
        .welcome-icon {
            width: 60px !important;
            height: 60px !important;
            font-size: 30px !important;
        }
        
        .welcome-title {
            font-size: 24px !important;
        }
    }
    """
    
    with gr.Blocks(css=custom_css, title="Kansas DMV Assistant") as app:
        # State variables
        show_welcome = gr.State(True)
        current_language = gr.State("en")
        
        with gr.Column(elem_classes=["main-container"]):
            # Header
            with gr.Row(elem_classes=["dmv-header"]):
                with gr.Column():
                    gr.HTML("""
                    <div class="header-left">
                        <div class="ks-logo">KS</div>
                        <div class="title-section">
                            <h1>Kansas DMV Assistant</h1>
                            <p>Your 24/7 Motor Vehicle Services Helper</p>
                        </div>
                    </div>
                    """)
                with gr.Column():
                    with gr.Row():
                        lang_display_btn = gr.Button("🌐 Spanish", elem_classes=["glass-btn"], scale=1)
                        gr.Button("ℹ️ Capabilities", elem_classes=["glass-btn"], scale=1)
            
            # Chat Container
            with gr.Column(elem_classes=["chat-container"]):
                # Welcome Screen
                welcome_html = gr.HTML("""
                <div class="welcome-screen" id="welcome-screen">
                    <h2 class="welcome-title">Welcome to Kansas DMV Assistant</h2>
                    <p class="welcome-subtitle">Get instant help with licenses, registrations, renewals, and more. Ask me anything about Kansas motor vehicle services!</p>
                    
                    <div class="prompts-grid">
                        <div class="prompt-card" onclick="document.getElementById('prompt-btn-0').click()">
                            <h4>🪪 License Renewal</h4>
                            <p>How do I renew my driver's license in Kansas?</p>
                        </div>
                        <div class="prompt-card" onclick="document.getElementById('prompt-btn-1').click()">
                            <h4>🚙 Vehicle Registration</h4>
                            <p>What documents do I need to register a new vehicle?</p>
                        </div>
                        <div class="prompt-card" onclick="document.getElementById('prompt-btn-2').click()">
                            <h4>💰 Fees & Costs</h4>
                            <p>How much does it cost to get a Kansas ID card?</p>
                        </div>
                        <div class="prompt-card" onclick="document.getElementById('prompt-btn-3').click()">
                            <h4>📍 Office Locations</h4>
                            <p>Where is the nearest DMV office to me?</p>
                        </div>
                        <div class="prompt-card" onclick="document.getElementById('prompt-btn-4').click()">
                            <h4>🚛 Commercial License</h4>
                            <p>What are the requirements for a CDL in Kansas?</p>
                        </div>
                        <div class="prompt-card" onclick="document.getElementById('prompt-btn-5').click()">
                            <h4>📅 Appointments</h4>
                            <p>How do I schedule a driving test appointment?</p>
                        </div>
                    </div>
                </div>
                """, visible=True)
                
                # Chatbot
                chatbot = gr.Chatbot(
                    elem_classes=["chatbot"],
                    height=400,
                    show_copy_button=True,
                    avatar_images=("👤", "🚗"),
                    visible=False
                )
            
            # Input Area
            with gr.Row(elem_classes=["input-container"]):
                msg = gr.Textbox(
                    placeholder="Ask me about licenses, registrations, renewals, fees, office locations...",
                    elem_classes=["input-box"],
                    container=False,
                    scale=4,
                    lines=1,
                    max_lines=3
                )
                send_btn = gr.Button("➤", elem_classes=["send-button"], scale=0, min_width=50)
        
        # Hidden buttons for sample prompts
        sample_prompts = [
            "How do I renew my driver's license in Kansas?",
            "What documents do I need to register a new vehicle?",
            "How much does it cost to get a Kansas ID card?",
            "Where is the nearest DMV office to me?",
            "What are the requirements for a CDL in Kansas?",
            "How do I schedule a driving test appointment?"
        ]
        
        prompt_buttons = []
        for i, prompt in enumerate(sample_prompts):
            btn = gr.Button(f"Prompt {i}", visible=False, elem_id=f"prompt-btn-{i}")
            prompt_buttons.append(btn)
        
        # Event handlers
        def handle_message(message, history, welcome_visible, language):
            if message.strip():
                # Process the message and hide welcome screen
                new_msg, new_history = chat_response(message, history, language)
                return new_msg, new_history, gr.update(visible=False), gr.update(visible=True), False
            return "", history, welcome_html, chatbot, welcome_visible
        
        def handle_prompt_click(prompt_text, history, language):
            # Process the prompt and hide welcome screen
            new_msg, new_history = chat_response(prompt_text, history, language)
            return "", new_history, gr.update(visible=False), gr.update(visible=True)
        
        def toggle_language(current_lang):
            """Toggle between English and Spanish."""
            if current_lang == "en":
                new_lang = "es"
                button_text = "🌐 Inglesa"
            else:
                new_lang = "en"
                button_text = "🌐 Spanish"
            
            return new_lang, button_text
        
        # Bind events
        msg.submit(
            handle_message,
            inputs=[msg, chatbot, show_welcome, current_language],
            outputs=[msg, chatbot, welcome_html, chatbot, show_welcome]
        )
        
        send_btn.click(
            handle_message,
            inputs=[msg, chatbot, show_welcome, current_language],
            outputs=[msg, chatbot, welcome_html, chatbot, show_welcome]
        )
        
        # Language toggle event
        lang_display_btn.click(
            toggle_language,
            inputs=[current_language],
            outputs=[current_language, lang_display_btn]
        )
        
        # Bind prompt button events
        for i, (btn, prompt) in enumerate(zip(prompt_buttons, sample_prompts)):
            btn.click(
                handle_prompt_click,
                inputs=[gr.State(prompt), chatbot, current_language],
                outputs=[msg, chatbot, welcome_html, chatbot]
            )
    
    return app

if __name__ == "__main__":
    print("\n🚀 Starting Kansas DMV Assistant...")
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