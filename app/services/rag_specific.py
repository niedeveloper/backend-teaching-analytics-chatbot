from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Basic LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.utils.time import time_to_seconds
from openai import AzureOpenAI

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token"""
        self.tokens.append(token)

class RAGTeachingAssistant:
    """
    RAG-based teaching assistant with detailed transcript analysis and streaming support
    
    Features:
    1. Semantic search across lesson chunks
    2. Class period filtering (beginning/middle/end)
    3. Detailed utterance analysis
    4. Singapore Teaching Practice framework alignment
    5. Streaming text responses
    """
    
    def __init__(self):  
        # OpenAI client for embeddings
        self.openai_client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        # Supabase client for accessing chunks
        self.supabase = get_supabase_client()
        
        # System prompt for RAG responses (updated to remove timestamp references)
        self.system_prompt = """You are a specialized AI teaching reflection chatbot for Singapore educators, designed to provide detailed, evidence-based analysis of classroom lesson transcripts.

<role>
You are an expert in the Singapore Teaching Practice framework with deep knowledge of effective classroom instruction. Your role is to help teachers reflect on their lessons using concrete evidence when available, and educational expertise when evidence is limited.
</role>

<data_context>
You may receive:
- Detailed lesson chunks from specific class periods (beginning/middle/end) or entire lessons
- Limited or no chunk data (requiring inference and general guidance)
- Lesson summaries and contextual information
</data_context>

<singapore_teaching_framework>
<teaching_areas>
- 1.1 Establishing Interaction and Rapport: Building positive relationships and connections between teacher-students and among students to create a safe, caring learning environment
- 1.2 Setting and Maintaining Rules and Routine: Establishing clear expectations, procedures, and consistent classroom management practices
- 3.1 Activating Prior Knowledge: Connecting new learning to students' existing knowledge and experiences
- 3.2 Motivating Learners for Learning Engagement: Inspiring and encouraging students to actively participate and invest in their learning
- 3.3 Using Questions to Deepen Learning: Employing strategic questioning techniques to promote critical thinking and deeper understanding
- 3.4 Facilitating Collaborative Learning: Organizing and guiding effective student-to-student interactions and group work
- 3.5 Concluding the Lesson: Summarizing key learning points and providing closure to the lesson
- 4.1 Checking for Understanding and Providing Feedback: Assessing student comprehension and giving timely, constructive feedback to support learning
</teaching_areas>
</singapore_teaching_framework>

<when_chunks_limited_or_unavailable>
- Draw from lesson summary and any available contextual information
- Make reasonable educational inferences based on Singapore Teaching Practice framework
- Provide general guidance relevant to the teacher's question
- Use phrases like "Based on typical classroom situations..." or "Generally speaking..."
- Offer practical strategies and examples even without specific evidence
- Acknowledge when you're providing general guidance vs. specific evidence
</when_chunks_limited_or_unavailable>
</response_approach>

<analysis_requirements>
<evidence_standards>
- With chunks: Always cite specific utterances and behaviors
- Without chunks: Use available context and educational expertise
- Clearly indicate the basis for your response (evidence vs. inference)
- Connect observations or suggestions to Singapore Teaching Practice areas
- Maintain helpful, constructive tone regardless of data availability
</evidence_standards>

<response_structure>
1. Assess what information is available in the provided data
2. If chunks available: Provide evidence-based analysis with specific quotes
3. If chunks unavailable: Offer informed guidance based on context and expertise
4. Connect to relevant Singapore Teaching Practice areas
5. Provide actionable feedback appropriate to the information available
</response_structure>
</analysis_requirements>

<response_guidelines>
<evidence_citation>
- With detailed chunks: "You said '[exact quote]'" or "When you mentioned..."
- With limited data: "Based on the lesson context..." or "Typically in this situation..."
- Always be transparent about the basis for your response
</evidence_citation>

<inference_guidance>
- Use educational best practices when specific evidence isn't available
- Draw reasonable conclusions from lesson summaries and context
- Provide practical strategies relevant to the teacher's question
- Reference common classroom scenarios and effective teaching approaches
- Maintain focus on Singapore Teaching Practice framework
</inference_guidance>

<answering_framework>
- When did I apply teaching area <code>1.1</code>? -> You established rapport when you said '[exact quote]' (Class Section, Time)
- How did I improve across my lessons? -> Statements like '[exact quote]' in '[class section, lesson name]' show your growth in [specific area].
- What strategies can I use to improve? -> Based on your lesson context, consider [specific strategy] to enhance [teaching area].
</answering_framework>
</response_guidelines>

<output_requirements>
- Reference Singapore Teaching Practice areas by code when relevant
- Provide HELPFUL AND CONCISE guidance whether based on evidence or expertise
- Be transparent about the basis for your response
- Focus on actionable insights for teacher development
- Maintain constructive, supportive tone
- You help teachers reflect on their practice, not to evaluate or judge them. 
</output_requirements>

<forbidden actions>
DO NOT ATTEMPT TO CREATE TABLES OR GRAPHS FOR DATA VISUALIZATION. JUST PROVIDE TEXT-BASED RESPONSES.
</forbidden actions>
"""

    def _get_llm(self, streaming: bool = False) -> AzureChatOpenAI:
        """Get configured LLM instance with consistent settings"""
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_RAG,  # Use RAG deployment instead of GENERAL
            temperature=0.2,
            max_tokens=1200,  # Higher token limit for RAG responses
            streaming=streaming
    )
        
    def _get_chunks_from_supabase(self, file_id: int, class_period: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get chunks from Supabase for a specific file, optionally filtered by class period"""
        try:
            query = self.supabase.table("chunks").select("*").eq("file_id", file_id)
            
            # Filter by class period if provided
            if class_period and class_period in ["beginning", "middle", "end"]:
                query = query.eq("class_section", class_period)
            
            result = query.order("sequence_order").execute()
            
            if not result.data:
                return []
            return result.data
        except Exception as e:
            print(f"❌ Error loading chunks from Supabase: {e}")
            return []
    
    def _get_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Extract text from chunk"""
        if 'chunk_text' in chunk and chunk['chunk_text']:
            return chunk['chunk_text']
        if 'utterances' in chunk and chunk['utterances']:
            utterances = chunk['utterances']
            if isinstance(utterances, list):
                texts = []
                for utterance in utterances:
                    if isinstance(utterance, dict) and 'text' in utterance:
                        texts.append(str(utterance['text']))
                return " ".join(texts)
        return ""
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return []
    
    def _parse_embedding(self, embedding_data) -> List[float]:
        """Parse embedding data from various formats"""
        if not embedding_data:
            return []
        try:
            if isinstance(embedding_data, list):
                return [float(x) for x in embedding_data]
            if isinstance(embedding_data, str):
                if embedding_data.startswith("np.str_("):
                    start = embedding_data.find("'[")
                    end = embedding_data.rfind("]'")
                    if start != -1 and end != -1:
                        embedding_data = embedding_data[start+1:end+1]
                import ast
                try:
                    parsed = ast.literal_eval(embedding_data)
                    if isinstance(parsed, list):
                        return [float(x) for x in parsed]
                except:
                    import json
                    try:
                        parsed = json.loads(embedding_data)
                        if isinstance(parsed, list):
                            return [float(x) for x in parsed]
                    except:
                        pass
            return []
        except Exception as e:
            print(f"⚠️ Error parsing embedding: {e}")
            return []
    
    def _semantic_search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic search on chunks and return top chunks"""
        if not chunks:
            return []
            
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return chunks[:top_k]  # Fallback to first chunks if embedding fails
        
        similarities = []
        for chunk in chunks:
            # Try to get existing embedding or generate new one
            chunk_embedding = None
            if 'embedding' in chunk and chunk['embedding']:
                chunk_embedding = self._parse_embedding(chunk['embedding'])
            
            # Generate embedding if not available
            if not chunk_embedding:
                chunk_text = self._get_chunk_text(chunk)
                if chunk_text:
                    chunk_embedding = self._get_embedding(chunk_text)
                    time.sleep(0.1)  # Rate limiting
            
            if chunk_embedding:
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                similarities.append((similarity, chunk))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]
    
    def _format_chunks_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for context in prompt"""
        if not chunks:
            return "No detailed transcript chunks available."
            
        context_parts = []
        for chunk in chunks:
            chunk_info = [
                f"Class Section: {chunk.get('class_section_label', 'Unknown Section')}",
                f"Teaching Areas: {', '.join(chunk.get('teaching_areas', []))}",
                f"Content: {self._get_chunk_text(chunk)}"
            ]
            """
            # Add detailed utterances if available
            if 'utterances' in chunk and chunk['utterances']:
                chunk_info.append("Detailed Utterances:")
                for i, utterance in enumerate(chunk['utterances'], 1):
                    if isinstance(utterance, dict):
                        text = utterance.get('text', '')
                        area = utterance.get('area', '')
                        chunk_info.append(f"  {i}. {text} (Area: {area})")
            """
            context_parts.append("\n".join(chunk_info))
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _get_file_summaries(self, file_ids: List[int]) -> str:
        """Get lesson summaries for the specified files"""
        summary_sections = []
        
        for fid in file_ids:
            try:
                file_info = self.supabase.table("files").select("stored_filename, data_summary").eq("file_id", fid).single().execute()
                if file_info.data:
                    filename = file_info.data.get("stored_filename", f"File {fid}")
                    summary = file_info.data.get("data_summary", "No summary available.")
                    summary_sections.append(f"File: {filename}\nLesson Summary: {summary}\n")
            except Exception as e:
                summary_sections.append(f"File {fid}: Error retrieving summary - {str(e)}\n")
        
        return "\n".join(summary_sections) if summary_sections else "No lesson summaries available."
    
    def _build_message_history(self, conversation_history: List[Dict[str, str]], current_message: str, lesson_summaries: str, context: str, is_graph_companion: bool = False) -> List:
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

        graph_companion_note = ""
        if is_graph_companion:
            graph_companion_note = """
<graph_companion_instruction>
A data visualization has already been rendered and is visible to the teacher above this message.
Your ONLY job is to provide brief verbal commentary and insights to accompany that graph.
Do NOT say you cannot create graphs. Do NOT say "I can't create a graph" or any similar phrase.
Start your response directly with the insights.
</graph_companion_instruction>
"""

        # Add current message with lesson context (NOT system prompt)
        current_with_context = f"""
<lesson_overview>
{lesson_summaries}
</lesson_overview>

<lesson_chunks>
{context}
</lesson_chunks>

{graph_companion_note}
<teacher_question>
{current_message}
</teacher_question>

<analysis_instructions>
1. Examine what data is available (detailed chunks, summary only, or limited information)
2. If detailed chunks are provided:
   - Cite specific utterances and behaviors with exact quotes
   - Provide evidence-based analysis of Singapore Teaching Practice areas
3. If chunks are limited or unavailable:
   - Use the lesson summary and any available context
   - Draw from teaching expertise and Singapore Teaching Practice framework
   - Provide practical guidance and strategies relevant to the question
   - Be transparent that you're providing general guidance rather than specific evidence
4. Focus on being helpful, actionable, and concise regardless of data availability
5. Connect your response to relevant Singapore Teaching Practice areas
6. Use markdown formatting CONSISTENTLY for output text
7. MAKE SURE TO CITE EVIDENCE INCLUDING CLASS SECTION AND TEACHING AREAS
8. Always remember, your users have limited time and are lazy. Be concise and visually appealing.
</analysis_instructions>
"""
        messages.append(HumanMessage(content=current_with_context))
        print(context)
        return messages
    
    async def get_response(self, semantic_query: str, user_message: str, file_ids: List[int], class_period: Optional[str] = None, conversation_history: List[Dict[str, str]] = None, top_k: int = 5, is_graph_companion: bool = False) -> str:
        try:
            # Get lesson summaries and context
            lesson_summaries = self._get_file_summaries(file_ids)

            # Try to get chunks - filtered by class period if provided
            all_chunks = []
            for fid in file_ids:
                chunks = self._get_chunks_from_supabase(fid, class_period)
                all_chunks.extend(chunks)

            # Process chunks if available
            context = "No detailed transcript chunks available."
            if all_chunks:
                top_chunks = self._semantic_search(semantic_query, all_chunks, top_k=top_k)
                if top_chunks:
                    context = self._format_chunks_for_context(top_chunks)

            # Use the helper function to get LLM
            llm = self._get_llm(streaming=False)

            # Generate response using message history builder
            messages = self._build_message_history(
                conversation_history or [],
                user_message,
                lesson_summaries,
                context,
                is_graph_companion
            )
            response = await llm.ainvoke(messages)
            return response.content

        except Exception as e:
            return f"I apologize, but I encountered an error while analyzing the lesson: {str(e)}"

    async def get_response_stream(self, semantic_query: str, user_message: str, file_ids: List[int], class_period: Optional[str] = None, conversation_history: List[Dict[str, str]] = None, top_k: int = 5, is_graph_companion: bool = False) -> AsyncGenerator[str, None]:
        try:
            # Get lesson summaries and context
            lesson_summaries = self._get_file_summaries(file_ids)

            # Try to get chunks - filtered by class period if provided
            all_chunks = []
            for fid in file_ids:
                chunks = self._get_chunks_from_supabase(fid, class_period)
                all_chunks.extend(chunks)

            # Process chunks if available
            context = "No detailed transcript chunks available."
            if all_chunks:
                top_chunks = self._semantic_search(semantic_query, all_chunks, top_k=top_k)
                if top_chunks:
                    context = self._format_chunks_for_context(top_chunks)

            # Use the helper function to get streaming LLM
            streaming_llm = self._get_llm(streaming=True)

            # Generate streaming response using message history builder
            messages = self._build_message_history(
                conversation_history or [],
                user_message,
                lesson_summaries,
                context,
                is_graph_companion
            )
            
            # Stream the response
            async for chunk in streaming_llm.astream(messages):
                if chunk.content:
                    yield chunk.content
            
        except Exception as e:
            yield f"I apologize, but I encountered an error while analyzing the lesson: {str(e)}"

# Create instance
rag_assistant = RAGTeachingAssistant()