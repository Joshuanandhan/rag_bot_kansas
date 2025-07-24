import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict, Any
import json
import math
import re

# Load environment variables from .env file
load_dotenv()

# Comprehensive Kansas DMV Office Database
KANSAS_DMV_OFFICES = [
    {
        "name": "Chanute Driver License Office",
        "address": "301 W. 14th St., Chanute, KS 66720",
        "phone": "620-431-7080",
        "email": "KDOR_ChanuteDL@KS.GOV",
        "latitude": 37.6756,
        "longitude": -95.4594,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Colby Driver License Office",
        "address": "990 S. Range St. #3, Colby, KS 67701",
        "phone": "785-462-3620",
        "email": "KDOR_ColbyDL@KS.GOV",
        "latitude": 39.3856,
        "longitude": -101.0537,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Dodge City Driver License Office",
        "address": "2601 Central Ave., Dodge City, KS 67801",
        "phone": "620-227-3944",
        "email": "KDOR_DodgeCityDL@KS.GOV",
        "latitude": 37.7528,
        "longitude": -100.0171,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Garden City Driver License Office",
        "address": "2506 N Johns St., Garden City, KS 67846",
        "phone": "620-276-8411",
        "email": "KDOR_GardenCityDL@KS.GOV",
        "latitude": 37.9917,
        "longitude": -100.8664,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Hays Driver License Office",
        "address": "1222 Canterbury Dr., Hays, KS 67601",
        "phone": "785-625-6917",
        "email": "KDOR_HaysDL@KS.GOV",
        "latitude": 38.8792,
        "longitude": -99.3267,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Phillipsburg Driver License Office",
        "address": "502 S. 7th St., Phillipsburg, KS 67661",
        "phone": "785-543-5594",
        "email": "KDOR_PhillipsburgDL@KS.GOV",
        "latitude": 39.7547,
        "longitude": -99.3267,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Salina Driver License Office",
        "address": "2941 Centennial Road, Salina, KS 67401",
        "phone": "785-825-0321",
        "email": "KDOR_SalinaDL@KS.GOV",
        "latitude": 38.8403,
        "longitude": -97.6114,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Seneca Driver License Office",
        "address": "203 N 8th St. Ste 2, Seneca, KS 66538",
        "phone": "785-336-6454",
        "email": "KDOR_SenecaDL@KS.GOV",
        "latitude": 39.8361,
        "longitude": -96.0636,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Topeka Driver License Office",
        "address": "3907 SW Burlingame Rd., Topeka, KS 66609",
        "phone": "785-266-8431",
        "email": "N/A",
        "latitude": 39.0119,
        "longitude": -95.7222,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"],
        "note": "Limited staff. Call before visiting for CDL written exams"
    },
    {
        "name": "Wichita Driver License Office",
        "address": "1873 W 21st N., Wichita, KS 67203",
        "phone": "785-940-1353",
        "email": "KDOR_WichitaDL@KS.GOV",
        "latitude": 37.6922,
        "longitude": -97.3375,
        "services": ["Driver's License", "CDL Testing", "ID Cards", "Written Tests", "Road Tests"]
    },
    {
        "name": "Edgerton CDL Testing Center",
        "address": "30750 W 193rd St., Edgerton, KS 66021",
        "phone": "785-581-2864",
        "email": "KDOR_EdgertonCDL@KS.GOV",
        "latitude": 38.7642,
        "longitude": -95.0122,
        "services": ["CDL Testing"],
        "note": "Limited staff. Call before visiting for CDL written exams"
    },
    {
        "name": "Lawrence County Treasurer",
        "address": "1006 Massachusetts St., Lawrence, KS 66044",
        "phone": "785-832-5263",
        "email": "treasurers@douglascountyks.org",
        "latitude": 38.9717,
        "longitude": -95.2353,
        "services": ["Vehicle Registration", "License Plates", "Title Services"]
    },
    {
        "name": "Manhattan County Treasurer",
        "address": "200 4th St., Manhattan, KS 66502",
        "phone": "785-537-6300",
        "email": "treasurer@rileycountyks.gov",
        "latitude": 39.1836,
        "longitude": -96.5717,
        "services": ["Vehicle Registration", "License Plates", "Title Services"]
    },
    {
        "name": "Kansas City County Treasurer",
        "address": "710 N 7th St., Kansas City, KS 66101",
        "phone": "913-573-8600",
        "email": "info@wycokck.org",
        "latitude": 39.1142,
        "longitude": -94.6275,
        "services": ["Vehicle Registration", "License Plates", "Title Services"]
    },
    {
        "name": "Overland Park County Treasurer",
        "address": "111 S Cherry St., Olathe, KS 66061",
        "phone": "913-715-0900",
        "email": "jocogov@jocogov.org",
        "latitude": 38.8814,
        "longitude": -94.8189,
        "services": ["Vehicle Registration", "License Plates", "Title Services"]
    },
    {
        "name": "Hutchinson County Treasurer",
        "address": "125 W 5th Ave., Hutchinson, KS 67501",
        "phone": "620-694-2624",
        "email": "treasurer@renoks.org",
        "latitude": 38.0608,
        "longitude": -97.9297,
        "services": ["Vehicle Registration", "License Plates", "Title Services"]
    }
]

class RAGAgent:
    def __init__(self, faiss_index_path: str = "unified_faiss_index"):
        """Initialize the RAG Agent with FAISS vectorstore and OpenAI models."""
        
        # Check if OpenAI API key is set in environment
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")
        
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
        
        # Create tools
        self.rag_tool = self._create_rag_tool()
        self.location_tool = self._create_location_tool()
        
        # Create memory for conversation
        self.memory = MemorySaver()
        
        # Create the agent
        self.agent_executor = create_react_agent(
            self.llm, 
            [self.rag_tool, self.location_tool], 
            checkpointer=self.memory
        )
        
        # System prompt to guide the agent
        self.system_prompt = """You are a helpful Kansas DMV assistant. You have access to two tools:

1. knowledge_base_search: Use this to search for specific information about Kansas driving laws, procedures, requirements, etc.

2. find_nearest_dmv: Use this AUTOMATICALLY when:
   - User asks about DMV office locations ("Where is the nearest DMV office?")
   - User's message contains location information (like "User's current location: 37.6922,-97.3375")
   - User asks about where to go for DMV services

IMPORTANT: If you see "User's current location:" in the message, ALWAYS use the find_nearest_dmv tool with that location information to provide nearby offices.

Be helpful, accurate, and use the tools to provide the best information possible."""
        
        print("🤖 RAG Agent initialized successfully with location services!")
    
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
    
    def _create_location_tool(self) -> Tool:
        """Create a location-based DMV office finder tool."""
        
        def find_nearest_dmv(location_query: str) -> str:
            """Find the nearest DMV office based on user's location."""
            try:
                # Extract coordinates from location query
                user_lat, user_lon = self._extract_coordinates(location_query)
                
                if user_lat is None or user_lon is None:
                    return f"Unable to determine location from: {location_query}. Please provide coordinates like '37.6922,-97.3375' or a city name."
                
                # Calculate distances to all DMV offices
                distances = []
                for office in KANSAS_DMV_OFFICES:
                    distance = self._calculate_distance(
                        user_lat, user_lon, 
                        office["latitude"], office["longitude"]
                    )
                    distances.append((distance, office))
                
                # Sort by distance and get the nearest 3
                distances.sort(key=lambda x: x[0])
                nearest_offices = distances[:3]
                
                # Format response
                response = "🏛️ **NEAREST KANSAS DMV OFFICES:**\n\n"
                
                for i, (distance, office) in enumerate(nearest_offices, 1):
                    response += f"**{i}. {office['name']}** ({distance:.1f} miles away)\n"
                    response += f"📍 Address: {office['address']}\n"
                    response += f"📞 Phone: {office['phone']}\n"
                    if office['email'] != "N/A":
                        response += f"📧 Email: {office['email']}\n"
                    response += f"🔧 Services: {', '.join(office['services'])}\n"
                    if office.get('note'):
                        response += f"⚠️ Note: {office['note']}\n"
                    response += "\n"
                
                response += "💡 **Tips:**\n"
                response += "• Call ahead to confirm hours and services\n"
                response += "• Check if you can schedule an appointment\n"
                response += "• Some services may be available online\n"
                response += "• Bring required documents (ID, proof of residency, etc.)\n"
                
                return response
                
            except Exception as e:
                return f"Error finding nearest DMV office: {str(e)}"
        
        return Tool(
            name="find_nearest_dmv",
            description="Find the nearest Kansas DMV office to the user's location. Use this tool AUTOMATICALLY when you see location information in the user's message (like 'User's current location: coordinates') or when users ask about DMV office locations. Input should be the user's location (coordinates or city name).",
            func=find_nearest_dmv
        )
    
    def _extract_coordinates(self, location_query: str) -> tuple:
        """Extract latitude and longitude from location query."""
        # Look for coordinate pattern like "37.6922,-97.3375"
        coord_pattern = r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)'
        match = re.search(coord_pattern, location_query)
        
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return lat, lon
        
        # If no coordinates found, try to match city names to approximate coordinates
        city_coords = {
            "wichita": (37.6922, -97.3375),
            "kansas city": (39.1142, -94.6275),
            "topeka": (39.0119, -95.7222),
            "overland park": (38.8814, -94.8189),
            "lawrence": (38.9717, -95.2353),
            "salina": (38.8403, -97.6114),
            "manhattan": (39.1836, -96.5717),
            "hutchinson": (38.0608, -97.9297),
            "hays": (38.8792, -99.3267),
            "dodge city": (37.7528, -100.0171),
            "garden city": (37.9917, -100.8664),
            "chanute": (37.6756, -95.4594),
            "colby": (39.3856, -101.0537)
        }
        
        location_lower = location_query.lower()
        for city, coords in city_coords.items():
            if city in location_lower:
                return coords
        
        return None, None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in miles
        r = 3956
        
        return c * r
    
    def chat(self, message: str, thread_id: str = "default") -> str:
        """Chat with the RAG agent."""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Check if this is a new conversation (no previous messages)
            state = self.agent_executor.get_state(config)
            existing_messages = state.values.get("messages", [])
            
            # Create messages list
            messages = []
            
            # Add system prompt if this is the first message in the conversation
            if not existing_messages:
                system_message = {"role": "system", "content": self.system_prompt}
                messages.append(system_message)
            
            # Add user message
            user_message = {"role": "user", "content": message}
            messages.append(user_message)
            
            # Get response from agent
            response = self.agent_executor.invoke(
                {"messages": messages}, 
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