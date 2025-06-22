import os
import getpass
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
# Removed unused import
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict, Any
import json

class RAGAgent:
    def __init__(self, faiss_index_path: str = "faiss_index"):
        """Initialize the RAG Agent with FAISS vectorstore and OpenAI models."""
        
        # Set up OpenAI API key if not already set
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Load the FAISS vectorstore
        try:
            self.vector_store = FAISS.load_local(
                faiss_index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Successfully loaded FAISS index from {faiss_index_path}")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            raise
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Create RAG tool
        self.rag_tool = self._create_rag_tool()
        
        # Create memory for conversation
        self.memory = MemorySaver()
        
        # Create the agent
        self.agent_executor = create_react_agent(
            self.llm, 
            [self.rag_tool], 
            checkpointer=self.memory
        )
        
        print("🤖 RAG Agent initialized successfully!")
    
    def _create_rag_tool(self) -> Tool:
        """Create a RAG tool that searches the vectorstore and returns relevant documents."""
        
        def rag_search(query: str) -> str:
            """Search the knowledge base for relevant information."""
            try:
                # Perform similarity search
                relevant_docs = self.vector_store.similarity_search(
                    query, 
                    k=5  # Return top 5 most relevant chunks
                )
                
                if not relevant_docs:
                    return "No relevant information found in the knowledge base."
                
                # Format the results
                context = []
                for i, doc in enumerate(relevant_docs, 1):
                    content = doc.page_content.strip()
                    metadata = doc.metadata
                    
                    # Add source information if available
                    source_info = ""
                    if metadata:
                        source_info = f" (Source: {metadata.get('source', 'Unknown')})"
                    
                    context.append(f"[Document {i}]{source_info}:\n{content}")
                
                return "\n\n".join(context)
                
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"
        
        return Tool(
            name="knowledge_base_search",
            description="Search the knowledge base for relevant information about Kansas driving, commercial driver's license, or motorcycle handbook content. Use this tool when you need to find specific information to answer user questions.",
            func=rag_search
        )
    
    def chat(self, message: str, thread_id: str = "default") -> str:
        """Chat with the RAG agent."""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Create input message
            input_message = {"role": "user", "content": message}
            
            # Get response from agent
            response = self.agent_executor.invoke(
                {"messages": [input_message]}, 
                config
            )
            
            # Extract the final AI message
            final_message = response["messages"][-1]
            return final_message.content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def stream_chat(self, message: str, thread_id: str = "default"):
        """Stream chat responses from the RAG agent."""
        
        config = {"configurable": {"thread_id": thread_id}}
        input_message = {"role": "user", "content": message}
        
        try:
            for step in self.agent_executor.stream(
                {"messages": [input_message]}, 
                config, 
                stream_mode="values"
            ):
                step["messages"][-1].pretty_print()
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def get_conversation_history(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        """Get the conversation history for a specific thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.agent_executor.get_state(config)
            
            messages = []
            for msg in state.values.get("messages", []):
                messages.append({
                    "role": msg.type,
                    "content": msg.content,
                    "timestamp": getattr(msg, "timestamp", None)
                })
            
            return messages
        except Exception as e:
            print(f"Error retrieving conversation history: {str(e)}")
            return []

def main():
    """Main function to run the RAG agent interactively."""
    
    try:
        # Initialize the RAG agent
        agent = RAGAgent()
        
        print("\n" + "="*60)
        print("🚗 Kansas Driving Knowledge Assistant")
        print("="*60)
        print("Ask me anything about Kansas driving regulations,")
        print("commercial driver's license, or motorcycle handbook!")
        print("Type 'quit' to exit, 'history' to see conversation history")
        print("="*60 + "\n")
        
        thread_id = "main_conversation"
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Drive safely!")
                    break
                
                if user_input.lower() == 'history':
                    history = agent.get_conversation_history(thread_id)
                    print("\n📜 Conversation History:")
                    for msg in history[-10:]:  # Show last 10 messages
                        role_emoji = "🧑" if msg["role"] == "human" else "🤖"
                        print(f"{role_emoji} {msg['role'].title()}: {msg['content'][:200]}...")
                    continue
                
                if not user_input:
                    continue
                
                print(f"\n🤖 Assistant:")
                # Use streaming for better user experience
                agent.stream_chat(user_input, thread_id)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Drive safely!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG agent: {str(e)}")

if __name__ == "__main__":
    main()