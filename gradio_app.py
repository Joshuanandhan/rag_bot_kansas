"""
Simple Gradio Web App for Kansas Driving Knowledge RAG Agent
"""
import gradio as gr
import os
from dotenv import load_dotenv
from agent import RAGAgent
from typing import List, Tuple
from deep_translator import GoogleTranslator
import logging
from datetime import datetime
import json
import time
import re
import uuid

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('gradio_app.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

def log_interaction(session_id: str, user_query: str, agent_response: str, language: str = "en", error: str = None):
    """
    Log user interactions with detailed information
    
    Args:
        session_id: Session identifier
        user_query: User's question
        agent_response: Agent's response
        language: Selected language
        error: Error message if any
    """
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'language': language,
        'user_query': user_query,
        'agent_response': agent_response[:200] + "..." if len(agent_response) > 200 else agent_response,
        'response_length': len(agent_response),
        'error': error
    }
    
    # Terminal output with colors
    print(f"\n{'='*80}")
    print(f"🕐 {datetime.now().strftime('%H:%M:%S')} | Session: {session_id}")
    print(f"🌐 Language: {language}")
    print(f"💬 Conversation Memory: ENABLED")
    print(f"{'='*80}")
    
    if error:
        print(f"❌ ERROR: {error}")
    else:
        print(f"👤 USER: {user_query}")
        print(f"{'─'*80}")
        print(f"🤖 AGENT: {agent_response}")
        print(f"{'─'*80}")
        print(f"📊 Response Length: {len(agent_response)} characters")
    
    print(f"{'='*80}\n")
    
    # File logging
    logger.info(json.dumps(log_data, ensure_ascii=False))

def generate_session_id() -> str:
    """Generate a unique session ID for the conversation."""
    return f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def translate_text(text, target_lang):
    """Translate text using Google Translator."""
    try:
        if target_lang == "en":
            return text
        
        logger.info(f"🔄 Translating to {target_lang}: {text[:100]}...")
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        logger.info(f"✅ Translation completed")
        return translated
    except Exception as e:
        logger.error(f"❌ Translation failed: {str(e)}")
        return text  # Return original if translation fails

def format_response_text(text: str) -> str:
    """
    Format response text to improve readability with proper HTML formatting
    
    Args:
        text: Raw response text
        
    Returns:
        Formatted HTML text
    """
    # Convert basic markdown-like formatting to HTML
    formatted = text
    
    # Convert **bold** to <strong>
    formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted)
    
    # Convert *italic* to <em>
    formatted = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted)
    
    # Convert ### Headers to <h3>
    formatted = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', formatted, flags=re.MULTILINE)
    
    # Convert ## Headers to <h2>
    formatted = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', formatted, flags=re.MULTILINE)
    
    # Convert # Headers to <h1>
    formatted = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', formatted, flags=re.MULTILINE)
    
    # Convert bullet points (- item) to <ul><li>
    lines = formatted.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f'<li>{stripped[2:]}</li>')
        elif stripped.startswith('• '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(line)
    
    if in_list:
        formatted_lines.append('</ul>')
    
    formatted = '\n'.join(formatted_lines)
    
    # Convert numbered lists (1. item) to <ol><li>
    formatted = re.sub(r'^\d+\.\s+(.*?)$', r'<li>\1</li>', formatted, flags=re.MULTILINE)
    
    # Wrap consecutive <li> items in <ol>
    formatted = re.sub(r'(<li>.*?</li>(?:\s*<li>.*?</li>)*)', r'<ol>\1</ol>', formatted, flags=re.DOTALL)
    
    # Convert double line breaks to paragraphs
    paragraphs = formatted.split('\n\n')
    formatted_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<'):
            formatted_paragraphs.append(f'<p>{para}</p>')
        else:
            formatted_paragraphs.append(para)
    
    formatted = '\n\n'.join(formatted_paragraphs)
    
    return formatted

# Global agent instance
agent = None

def initialize_agent():
    """Initialize the RAG agent."""
    global agent
    try:
        logger.info("🚀 Initializing RAG Agent...")
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            error_msg = "OPENAI_API_KEY not found. Please add it to your .env file."
            logger.error(f"❌ {error_msg}")
            return f"❌ Error: {error_msg}"
        
        if agent is None:
            agent = RAGAgent()
            logger.info("✅ RAG Agent initialized successfully with conversational memory")
        return "✅ Agent ready!"
    except Exception as e:
        error_msg = f"Failed to initialize agent: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return f"❌ Error: {error_msg}"

def simulate_streaming_response(response_text: str, history: List[Tuple[str, str]], user_message: str):
    """
    Simulate ChatGPT-like streaming response with proper formatting
    
    Args:
        response_text: Complete response text
        history: Chat history
        user_message: User's message
        
    Yields:
        Updated history with streaming response
    """
    # Add user message and empty assistant response
    history.append((user_message, ""))
    
    # Preserve formatting by splitting more carefully
    
    # Split text into tokens while preserving whitespace and newlines
    tokens = re.findall(r'\S+|\s+', response_text)
    
    current_response = ""
    
    for i, token in enumerate(tokens):
        current_response += token
        
        # Update the last message in history
        history[-1] = (user_message, current_response.rstrip())
        
        # Add delay only for actual words, not whitespace
        if token.strip():  # Only delay for non-whitespace tokens
            time.sleep(0.05)
        
        yield history

def chat_response_streaming(message: str, history: List[Tuple[str, str]], language: str, session_id: str):
    """Process chat message with streaming response and persistent session."""
    global agent
    
    if not message.strip():
        return history
    
    # Initialize agent if needed
    if agent is None:
        logger.info("🔄 Agent not initialized, initializing now...")
        init_result = initialize_agent()
        if "Error" in init_result:
            log_interaction(session_id, message, init_result, language, error=init_result)
            history.append((message, init_result))
            yield history
            return
    
    try:
        logger.info(f"🔄 Processing query for session {session_id} (Memory: ENABLED)")
        
        # Add user message with typing indicator
        history.append((message, "🤔 Thinking..."))
        yield history
        
        # Get response from agent with persistent session_id
        response = agent.chat(message, session_id)
        
        # Translate response if Spanish is selected
        if language == "es":
            response = translate_text(response, "es")
        
        # Format the response for better display
        formatted_response = format_response_text(response)
        
        # Log the interaction
        log_interaction(session_id, message, response, language)
        
        # Stream the formatted response
        for updated_history in simulate_streaming_response(formatted_response, history[:-1], message):
            yield updated_history
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(f"❌ {error_msg}")
        log_interaction(session_id, message, error_msg, language, error=error_msg)
        history[-1] = (message, f"❌ {error_msg}")
        yield history

def clear_conversation(session_id: str):
    """Clear the conversation history for the current session."""
    global agent
    try:
        if agent is not None:
            # Generate a new session ID to effectively clear the conversation
            new_session_id = generate_session_id()
            logger.info(f"🗑️ Conversation cleared. New session: {new_session_id}")
            return [], new_session_id, gr.update(visible=True), gr.update(visible=False)
        else:
            new_session_id = generate_session_id()
            return [], new_session_id, gr.update(visible=True), gr.update(visible=False)
    except Exception as e:
        logger.error(f"❌ Error clearing conversation: {str(e)}")
        return [], session_id, gr.update(visible=True), gr.update(visible=False)

def get_conversation_summary(session_id: str):
    """Get a summary of the current conversation."""
    global agent
    try:
        if agent is not None:
            history = agent.get_conversation_history(session_id)
            if history:
                message_count = len(history)
                user_messages = [msg for msg in history if msg['role'] == 'human']
                assistant_messages = [msg for msg in history if msg['role'] == 'assistant']
                
                summary = f"📊 **Conversation Summary (Session: {session_id})**\n\n"
                summary += f"• **Total Messages:** {message_count}\n"
                summary += f"• **User Messages:** {len(user_messages)}\n"
                summary += f"• **Assistant Messages:** {len(assistant_messages)}\n\n"
                
                if user_messages:
                    summary += "**Recent Topics:**\n"
                    recent_topics = [msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content'] 
                                   for msg in user_messages[-3:]]  # Last 3 user messages
                    for i, topic in enumerate(recent_topics, 1):
                        summary += f"{i}. {topic}\n"
                
                return summary
            else:
                return "No conversation history found."
        else:
            return "Agent not initialized."
    except Exception as e:
        logger.error(f"❌ Error getting conversation summary: {str(e)}")
        return f"Error retrieving conversation summary: {str(e)}"

def chat_response(message: str, history: List[Tuple[str, str]], language: str, session_id: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Process chat message and return response (fallback for non-streaming)."""
    global agent
    
    if not message.strip():
        return "", history
    
    # Initialize agent if needed
    if agent is None:
        logger.info("🔄 Agent not initialized, initializing now...")
        init_result = initialize_agent()
        if "Error" in init_result:
            log_interaction(session_id, message, init_result, language, error=init_result)
            history.append((message, init_result))
            return "", history
    
    try:
        logger.info(f"🔄 Processing query for session {session_id} (Memory: ENABLED)")
        
        # Get response from agent with persistent session_id
        response = agent.chat(message, session_id)
        
        # Translate response if Spanish is selected
        if language == "es":
            response = translate_text(response, "es")
        
        # Format the response for better display
        formatted_response = format_response_text(response)
        
        # Log the interaction
        log_interaction(session_id, message, response, language)
        
        history.append((message, formatted_response))
        return "", history
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(f"❌ {error_msg}")
        log_interaction(session_id, message, error_msg, language, error=error_msg)
        history.append((message, f"❌ {error_msg}"))
        return "", history

def create_app():
    """Create the modern ChatGPT-inspired interface."""
    
    logger.info("🎨 Creating Gradio interface...")
    
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
    
    .clear-btn {
        background: rgba(255,87,87,0.2) !important;
        border: 1px solid rgba(255,87,87,0.3) !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    .clear-btn:hover {
        background: rgba(255,87,87,0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    .info-btn {
        background: rgba(52,152,219,0.2) !important;
        border: 1px solid rgba(52,152,219,0.3) !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    .info-btn:hover {
        background: rgba(52,152,219,0.3) !important;
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
    
    .memory-indicator {
        font-size: 14px !important;
        color: #27ae60 !important;
        margin-bottom: 20px !important;
        font-weight: bold !important;
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
    
    /* Chat message formatting */
    .chatbot .message {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        line-height: 1.6 !important;
        font-size: 14px !important;
    }
    
    .chatbot .message p {
        margin: 0 0 12px 0 !important;
        line-height: 1.6 !important;
    }
    
    .chatbot .message p:last-child {
        margin-bottom: 0 !important;
    }
    
    /* Format lists and bullet points */
    .chatbot .message ul,
    .chatbot .message ol {
        margin: 8px 0 !important;
        padding-left: 20px !important;
    }
    
    .chatbot .message li {
        margin: 4px 0 !important;
        line-height: 1.5 !important;
    }
    
    /* Format headings */
    .chatbot .message h1,
    .chatbot .message h2,
    .chatbot .message h3,
    .chatbot .message h4,
    .chatbot .message h5,
    .chatbot .message h6 {
        margin: 16px 0 8px 0 !important;
        font-weight: bold !important;
    }
    
    /* Format code blocks */
    .chatbot .message code {
        background: #f4f4f4 !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .chatbot .message pre {
        background: #f8f9fa !important;
        padding: 12px !important;
        border-radius: 6px !important;
        overflow-x: auto !important;
        border-left: 4px solid #667eea !important;
        margin: 8px 0 !important;
    }
    
    /* Format blockquotes */
    .chatbot .message blockquote {
        border-left: 4px solid #667eea !important;
        padding-left: 16px !important;
        margin: 8px 0 !important;
        font-style: italic !important;
        color: #555 !important;
    }
    
    /* Format tables */
    .chatbot .message table {
        border-collapse: collapse !important;
        width: 100% !important;
        margin: 8px 0 !important;
    }
    
    .chatbot .message th,
    .chatbot .message td {
        border: 1px solid #ddd !important;
        padding: 8px 12px !important;
        text-align: left !important;
    }
    
    .chatbot .message th {
        background: #f8f9fa !important;
        font-weight: bold !important;
    }
    
    /* Format strong and emphasis */
    .chatbot .message strong,
    .chatbot .message b {
        font-weight: bold !important;
    }
    
    .chatbot .message em,
    .chatbot .message i {
        font-style: italic !important;
    }
    
    /* Format links */
    .chatbot .message a {
        color: #667eea !important;
        text-decoration: underline !important;
    }
    
    .chatbot .message a:hover {
        color: #5a67d8 !important;
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
        logger.info("🎭 Setting up Gradio blocks...")
        
        # State variables
        show_welcome = gr.State(True)
        current_language = gr.State("en")
        session_id = gr.State(generate_session_id())  # Persistent session ID
        
        with gr.Column(elem_classes=["main-container"]):
            # Header
            with gr.Row(elem_classes=["dmv-header"]):
                with gr.Column():
                    gr.HTML("""
                    <div class="header-left">
                        <div class="ks-logo">KS</div>
                        <div class="title-section">
                            <h1>Kansas DMV Assistant</h1>
                            <p>Your 24/7 Motor Vehicle Services Helper with Memory</p>
                        </div>
                    </div>
                    """)
                with gr.Column():
                    with gr.Row():
                        lang_display_btn = gr.Button("🌐 Spanish", elem_classes=["glass-btn"], scale=1)
                        clear_btn = gr.Button("🗑️ Clear", elem_classes=["clear-btn"], scale=1)
                        info_btn = gr.Button("📊 Info", elem_classes=["info-btn"], scale=1)
                    
            
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
                    avatar_images=("🙋‍♂️", "🏛️"),
                    visible=False,
                    sanitize_html=False,
                    render_markdown=True
                )
                
                # Info display for conversation summary
                info_display = gr.Markdown(visible=False)
            
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
        def handle_message_streaming(message, history, language, session_id):
            """Handle message with ChatGPT-like streaming"""
            if message.strip():
                logger.info(f"📝 New message received: {message[:50]}... (Session: {session_id})")
                # Process with streaming
                for updated_history in chat_response_streaming(message, history, language, session_id):
                    yield "", updated_history, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def handle_message(message, history, welcome_visible, language, session_id):
            if message.strip():
                logger.info(f"📝 New message received: {message[:50]}... (Session: {session_id})")
                # Process the message and hide welcome screen
                new_msg, new_history = chat_response(message, history, language, session_id)
                return new_msg, new_history, gr.update(visible=False), gr.update(visible=True), False, gr.update(visible=False)
            return "", history, welcome_html, chatbot, welcome_visible, gr.update(visible=False)
        
        def handle_prompt_click(prompt_text, history, language, session_id):
            logger.info(f"🎯 Sample prompt clicked: {prompt_text} (Session: {session_id})")
            # Process the prompt with streaming
            for updated_history in chat_response_streaming(prompt_text, history, language, session_id):
                yield "", updated_history, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def toggle_language(current_lang):
            """Toggle between English and Spanish."""
            if current_lang == "en":
                new_lang = "es"
                button_text = "🌐 English"
                logger.info("🌐 Language switched to Spanish")
            else:
                new_lang = "en"
                button_text = "🌐 Spanish"
                logger.info("🌐 Language switched to English")
            
            return new_lang, button_text
        
        def handle_clear_conversation(session_id):
            """Handle clearing the conversation."""
            return clear_conversation(session_id)
        
        def handle_info_click(session_id):
            """Handle info button click to show conversation summary."""
            summary = get_conversation_summary(session_id)
            return gr.update(value=summary, visible=True)
        
        def hide_info():
            """Hide the info display."""
            return gr.update(visible=False)
        
        # Bind events with streaming
        msg.submit(
            handle_message_streaming,
            inputs=[msg, chatbot, current_language, session_id],
            outputs=[msg, chatbot, welcome_html, chatbot, info_display]
        )
        
        send_btn.click(
            handle_message_streaming,
            inputs=[msg, chatbot, current_language, session_id],
            outputs=[msg, chatbot, welcome_html, chatbot, info_display]
        )
        
        # Language toggle event
        lang_display_btn.click(
            toggle_language,
            inputs=[current_language],
            outputs=[current_language, lang_display_btn]
        )
        
        # Clear conversation event
        clear_btn.click(
            handle_clear_conversation,
            inputs=[session_id],
            outputs=[chatbot, session_id, welcome_html, chatbot]
        )
        
        # Info button event
        info_btn.click(
            handle_info_click,
            inputs=[session_id],
            outputs=[info_display]
        )
        
        # Hide info when clicking elsewhere
        chatbot.change(
            hide_info,
            outputs=[info_display]
        )
        
        # Bind prompt button events with streaming
        for i, (btn, prompt) in enumerate(zip(prompt_buttons, sample_prompts)):
            btn.click(
                handle_prompt_click,
                inputs=[gr.State(prompt), chatbot, current_language, session_id],
                outputs=[msg, chatbot, welcome_html, chatbot, info_display]
            )
    
    logger.info("✅ Gradio interface created successfully with conversational memory")
    return app

if __name__ == "__main__":
    print("\n🚀 Starting Kansas DMV Assistant...")
    print("📍 Initializing RAG Agent...")
    
    # Pre-initialize the agent
    init_status = initialize_agent()
    print(f"   {init_status}")
    
    print("\n🌐 Launching web interface...")
    print("   Access at: http://localhost:7860")
    print("   Press Ctrl+C to stop")
    print("\n💬 NEW FEATURES:")
    print("   ✅ Conversational Memory - Agent remembers the conversation!")
    print("   ✅ Clear Conversation - Reset conversation history")
    print("   ✅ Conversation Info - View conversation summary")
    print("   ✅ Persistent Session - Same session across all messages")
    print("\n📝 LIVE LOGS - User interactions will appear below:")
    print("="*80)
    
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
        logger.info("🛑 Application stopped by user")