"""
Example script demonstrating how to use the RAG Agent
"""
from agent import RAGAgent

def run_examples():
    """Run some example queries with the RAG agent."""
    
    # Initialize the agent
    print("Initializing RAG Agent...")
    agent = RAGAgent()
    
    # Example queries
    example_questions = [
        "What are the requirements to get a commercial driver's license in Kansas?",
        "What are the age requirements for a motorcycle license?",
        "What is the maximum speed limit on Kansas highways?",
        "What are the penalties for DUI in Kansas?",
        "What documents do I need to bring for a driving test?"
    ]
    
    print("\n" + "="*60)
    print("🚗 RAG Agent Example Queries")
    print("="*60)
    
    thread_id = "example_session"
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 50)
        
        try:
            # Get response from the agent
            response = agent.chat(question, thread_id)
            print(f"🤖 Answer: {response}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 50)
    
    print("\n✅ Example queries completed!")

def interactive_mode():
    """Run the agent in interactive mode."""
    
    agent = RAGAgent()
    
    print("\n" + "="*60)
    print("🚗 Interactive RAG Agent")
    print("="*60)
    print("Ask me anything about Kansas driving!")
    print("Type 'quit' to exit")
    print("="*60)
    
    thread_id = "interactive_session"
    
    while True:
        try:
            question = input("\n🧑 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🤖 Assistant:")
            agent.stream_chat(question, thread_id)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        run_examples()
    else:
        interactive_mode()
