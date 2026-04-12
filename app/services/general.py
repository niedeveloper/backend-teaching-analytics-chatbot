from typing import List, Dict, Any, Optional, AsyncGenerator 
import json

# Basic LangChain imports - only the essentials
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.db.supabase import get_supabase_client

class GeneralTeachingAssistant:
    """
    General teaching assistant with lesson summary context
    
    Features:
    1. Access to lesson summaries for context
    2. General teaching advice and guidance
    3. Singapore Teaching Practice framework awareness
    4. Conversational teaching support
    """
    
    def __init__(self):
        # Supabase client for accessing file summaries
        self.supabase = get_supabase_client()
        
        # System prompt for general teaching assistance
        self.system_prompt = """You are a general teaching chatbot assistant designed to provide helpful, supportive guidance to Singaporean teachers based on their lesson context and questions according to the Singapore Teaching Framework.

<singapore_teaching_framework>
Reference these EXACT WORDING of these areas (INCLUDING THE NUMBERS) when providing feedback:
- 1.1 Establishing Interaction and Rapport
- 1.2 Setting and Maintaining Rules and Routine
- 3.1 Activating Prior Knowledge
- 3.2 Motivating Learners for Learning Engagement
- 3.3 Using Questions to Deepen Learning
- 3.4 Facilitating Collaborative Learning
- 3.5 Concluding the Lesson
- 4.1 Checking for Understanding and Providing Feedback
</singapore_teaching_framework>

<forbidden_actions>
When users ask about anything unrelated to teaching, lesson summaries, or general teaching advice.
DO NOT generate data visualizations, charts, or tables yourself. When a user asks for a graph or chart,
a visualization is already being rendered and displayed to the teacher separately by the system.
Your role is to provide ONLY text-based insights and commentary about what the data shows.
NEVER say "I'm unable to create graphs" or similar refusals — treat the graph as already visible to
the teacher and provide verbal analysis of the patterns and insights in the data.
</forbidden_actions>
"""
    
    def _get_llm(self, streaming: bool = False) -> AzureChatOpenAI:
        """Get configured LLM instance with consistent settings"""
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_GENERAL,
            temperature=0.2,  # Consistent temperature
            max_tokens=700,
            streaming=streaming
        )
    
    def _get_file_summaries(self, file_ids: List[int]) -> str:
        """Get lesson summaries for the specified files"""
        if not file_ids:
            return "No lesson summaries available."
            
        summary_sections = []
        
        for fid in file_ids:
            try:
                result = self.supabase.table("files").select("stored_filename, data_summary").eq("file_id", fid).single().execute()
                if result.data:
                    filename = result.data.get("stored_filename", f"File {fid}")
                    summary = result.data.get("data_summary", "No summary available.")
                    summary_sections.append(f"**{filename}**\n{summary}")
            except Exception as e:
                print(f"Error retrieving summary for file {fid}: {e}")  # Log for debugging
                summary_sections.append(f"File {fid}: Summary unavailable")
        
        return "\n\n".join(summary_sections) if summary_sections else "No lesson summaries available."
    
    def _build_message_history(self, conversation_history: List[Dict[str, str]], current_message: str, lesson_summaries: str) -> List:
        """Build proper message history for LangChain with lesson context"""
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add conversation history (keep last 6 messages for better context)
        for msg in conversation_history[-6:]:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        
        # Add current message with lesson context (NOT system prompt)
        current_with_context = f"""
<lesson_context>
{lesson_summaries}
</lesson_context>


<teacher_question>
{current_message}
</teacher_question>

<chatbot_guidelines>
You will be provided with lesson summaries. Use these conditionally when ONLY IF the teacher asks about it.
Be absolutely CONCISE AND DIRECT in your responses. Avoid unnecessary verbosity. Be specific ONLY IF teacher asks to be specific.
When asked about trends, summarize key points and insights about that trend unless prompted otherwise. 

<if asked about lesson summaries>
DO NOT provide the summaries directly. Instead, summarize key points and insights, the summary table will be displayed outside of your response. 
</chatbot_guidelines>
"""
        
        messages.append(HumanMessage(content=current_with_context))
        return messages
    
    async def get_response(self, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Get general teaching response with lesson context
        
        Args:
            user_message: The teacher's question or message
            file_ids: List of lesson file IDs for summary context
            conversation_history: Previous conversation context
            
        Returns:
            Text response with teaching guidance
        """
        try:
            # Get lesson summaries for context
            lesson_summaries = self._get_file_summaries(file_ids)
            # Use conversation history for continuity
            messages = self._build_message_history(conversation_history, user_message, lesson_summaries)
            response = await self._get_llm().ainvoke(messages)
            return response.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    async def get_response_stream(self, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """
        Stream general teaching response with lesson context
        
        Args:
            user_message: The teacher's question or message
            file_ids: List of lesson file IDs for summary context
            conversation_history: Previous conversation context
            
        Yields:
            Text chunks as they're generated
        """
        try:
            # Get lesson summaries for context
            lesson_summaries = self._get_file_summaries(file_ids)
            
            messages = self._build_message_history(conversation_history, user_message, lesson_summaries)
            
            # Stream the response
            streaming_llm = self._get_llm(streaming=True)
            async for chunk in streaming_llm.astream(messages):
                if chunk.content:
                    yield chunk.content

            
        except Exception as e:
            yield f"I apologize, but I encountered an error while processing your request: {str(e)}"

# Create instance
general_assistant = GeneralTeachingAssistant()