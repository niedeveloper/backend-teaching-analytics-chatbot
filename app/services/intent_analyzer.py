from typing import List, Dict, Any, Optional, AsyncGenerator 
import json

# Basic LangChain imports - only the essentials
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.services.graph_registry import get_graphs_for_intent_analysis, validate_graph_type, map_natural_language_to_area_codes, get_available_area_codes, AVAILABLE_GRAPHS

class IntentAnalyzer:
    """
    Intent Analyzer for routing queries between general and RAG assistants
    
    Features:
    1. Analyzes user queries to determine intent and routing needs
    2. Classifies queries as lesson-specific or general teaching questions
    3. Returns routing decision with confidence scores
    4. Singapore Teaching Practice framework awareness
    5. Considers lesson availability and relevance
    6. Detects lesson and teaching area filtering for graphs
    """
    
    def __init__(self):
        # Azure OpenAI connection
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_RAG,
            temperature=0.1,  # Low temperature for consistent routing decisions
            max_tokens=600
        )
        
        # Supabase client for accessing file summaries
        self.supabase = get_supabase_client()
        
        # System prompt for intent analysis and routing
        self.system_prompt = f"""You are an Intent Analyzer Agent for a Singapore educator teaching assistant system. Your primary responsibility is to analyze user questions and conversation history to determine the optimal tool combination for providing the best response.

<role>
You are a routing specialist that understands teaching contexts and can efficiently direct questions to the most appropriate analysis tools and agents based on the specificity and scope. Transform the user query into a better query that helps the following agents respond more effectively.

<transform_query>
Transform user query into a more effective query for the agents IF:
- User question is too vague
- User asks about data visualization

If the user requests a graph, chart, or visualization:
- Set needs_graph: true (the graph will be rendered separately by the system)
- Transform the query into commentary framing so the agent provides verbal analysis alongside the graph
- Examples: "Show me a graph of xxx" → "Comment on my xxx across the lesson"
           "Could you graph my teaching area distribution" → "Provide insights on my teaching area distribution across the lessons"
           "Can you chart my questioning patterns" → "Analyze my questioning patterns throughout the lesson"
- The downstream agent does NOT generate the graph — it only provides the text commentary to accompany it

CRITICAL SCOPE PRESERVATION RULES:
- DO NOT EXPAND THE SCOPE OF THE QUERY
- If user asks for "1.1", transformed_query must focus ONLY on 1.1
- If user asks for "3.4", transformed_query must focus ONLY on 3.4
- If user asks for specific area codes, do NOT add related areas
- Preserve the exact teaching areas the user specified
- Only expand scope if user explicitly asks for "all areas" or similar broad terms
</transform_query>

<forbidden_actions>
When you are not sure about the intent, do not make assumptions. Route to the general assistant for broad questions or when the intent is unclear.
If you are unsure on how to transform the query, return the original query as transformed_query. DO NOT attempt to modify it.
</forbidden_actions>

<available_tools>
<database_query>
Time Period Selection (choose 1 of 3):
- beginning: First 10 minutes of lesson
- middle1: Next 20 minutes of lesson
- middle2: Next 20 minutes of lesson after middle1
- end: Last 110 minutes of lesson

Usage: When user specifies or implies a specific time period in their question
</database_query>

<available_agents>
<general_assistant>
- Access: Lesson data summary only
- Best for: General feedback, broad lesson evaluation, teaching advice
- Use when: Questions can be answered with high-level lesson overview
</general_assistant>

<rag_assistant>
- Access: Lesson data summary + database retrieval + RAG of relevant chunks
- Best for: Specific examples, detailed analysis, evidence-based responses
- Use when: Questions require specific evidence, examples, or detailed transcript analysis
</available_agents>
</available_tools>

<area_filtering>
Analyze user query for teaching area focus with STRICT SCOPE PRESERVATION:

PRIORITY 1 - EXPLICIT AREA CODES: If user mentions specific area codes (e.g., "1.1", "3.4", "2.1"), use ONLY those codes and DO NOT expand to related areas.

PRIORITY 2 - NATURAL LANGUAGE MAPPING: Only apply broad natural language mapping when NO explicit area codes are mentioned:
- "focus on questioning", "just show questioning" → area_filter: ["3.3"]
- "interaction and collaboration" (when no specific codes mentioned) → area_filter: ["1.1", "3.4"]
- "motivation and engagement" (when no specific codes mentioned) → area_filter: ["3.2"]
- "all areas" or no specific area mentioned → area_filter: [] (show all areas)

CRITICAL: If user says "1.1" or "analyze 1.1", use area_filter: ["1.1"] ONLY. Do NOT expand to include 3.4 or any other areas.

Available teaching areas: {get_available_area_codes()}
</area_filtering>

<decision_framework>
<time_period_analysis>
Analyze user question for time-specific language:
- "beginning", "start", "opening", "first X minutes" → beginning
- "middle", "during", "in the middle", "around minute X" (where X is 15-30) → middle  
- "end", "conclusion", "closing", "last X minutes", "wrap up" → end
- No specific time mentioned → null (no database query needed)
</time_period_analysis>

<graph_detection>
Analyze if the user query would benefit from data visualization:

Graph indicators:
- "show me a chart", "visualize", "graph", "plot", "draw"
- "compare", "trend", "pattern", "distribution", "overview"
- "how often", "frequency", "over time", "across lessons"
- "breakdown", "proportion", "percentage", "statistics"
- "see the data", "visual representation", "chart of"

Available graphs:
{self._get_graph_descriptions_for_prompt()}

Usage: When user requests visual representation, charts, graphs, or data visualization

Choose the most appropriate graph(s) based on:
- Teaching area focus → teaching_area_distribution or total_distribution
- Time-based analysis → utterance_timeline or area_distribution_time
- Speaking pace analysis → wpm_trend
- General comparison → total_distribution
- Comprehensive analysis → multiple complementary graphs

MULTIPLE GRAPHS: When users ask for comprehensive views, comparisons, or multiple perspectives, provide multiple graphs that complement each other.
</graph_detection>

<agent_selection_criteria>
Choose RAG ASSISTANT when:
- User requests specific examples or evidence
- Questions about particular teaching moments or interactions
- Requests for detailed analysis of specific behaviors
- Questions that need transcript-level detail to answer properly
- Follow-up questions requesting specifics after general responses
- **Time-specific questions (beginning/middle/end) - these ALWAYS need RAG for detailed analysis**
DO NOT USE RAG ASSISTANT ALONG WITH GRAPH GENERATION

Choose GENERAL ASSISTANT when:
- User asks about data visualization
- Questions are broad and evaluative WITHOUT time specificity ("How did my lesson go overall?")
- Requests for general teaching advice or strategies
- High-level performance assessment questions that cover the ENTIRE lesson
- Questions answerable from lesson summary alone
- Social conversation or greetings
</agent_selection_criteria>

<conversation_history_considerations>
When conversation history is available:
- If user previously received general feedback and now asks for specifics → RAG Assistant
- If user asks follow-up questions like "show me examples" → RAG Assistant
- If user references specific moments mentioned in previous responses → RAG Assistant + appropriate time period
- If continuing general discussion → General Assistant
</conversation_history_considerations>
</decision_framework>

<analysis_process>
1. Examine user question for time-specific references
2. Determine if question needs general summary or detailed evidence
3. Consider conversation history for context escalation
4. Select appropriate time period (if applicable)
5. Choose optimal agent based on question specificity and data needs
6. Detect lesson filtering needs (specific lessons vs all lessons)
7. Detect area filtering needs (specific teaching areas vs all areas)
</analysis_process>

<decision_examples>
<general_assistant_examples>
- "How did my lesson go overall?" → agent: general_assistant, class_period: null, lesson_filter: [], area_filter: []
- "What teaching strategies worked well?" → agent: general_assistant, class_period: null, lesson_filter: [], area_filter: []
- "Can you give me feedback on my lesson?" → agent: general_assistant, class_period: null, lesson_filter: [], area_filter: []
</general_assistant_examples>

<rag_assistant_examples>
- "Show me examples of when I asked good questions" → agent: rag_assistant, class_period: null, lesson_filter: [], area_filter: []
- "What happened in the first 15 minutes?" → agent: rag_assistant, class_period: beginning, lesson_filter: [], area_filter: []
- "Can you find specific instances when students were engaged?" → agent: rag_assistant, class_period: null, lesson_filter: [], area_filter: []
- "How did I conclude the lesson?" → agent: rag_assistant, class_period: end, lesson_filter: [], area_filter: []
</rag_assistant_examples>

<graph_examples>
- "Show me a chart comparing lesson 1 and 2" → needs_graph: true, graph_types: [{{"type": "teaching_area_distribution", "reason": "Shows distribution comparison between lessons"}}], area_filter: []
- "Focus on my questioning patterns" → needs_graph: true, graph_types: [{{"type": "utterance_timeline", "reason": "Shows questioning patterns over time"}}], lesson_filter: [], area_filter: ["3.3"]
- "Compare my interaction skills across all lessons" → needs_graph: true, graph_types: [{{"type": "total_distribution", "reason": "Shows overall interaction skills distribution"}}], lesson_filter: [], area_filter: ["1.1"]
- "Show me speaking pace for the first lesson only" → needs_graph: true, graph_types: [{{"type": "wpm_trend", "reason": "Shows speaking pace over time for the first lesson"}}], lesson_filter: ["Science_Lesson1(09-07-2024).xlsx"], area_filter: []
- "Give me a comprehensive overview of my teaching" → needs_graph: true, graph_types: [{{"type": "total_distribution", "reason": "Shows overall teaching area distribution"}}, {{"type": "utterance_timeline", "reason": "Shows teaching patterns over time"}}], lesson_filter: [], area_filter: []
- "Show me both the area distribution and my questioning patterns" → needs_graph: true, graph_types: [{{"type": "teaching_area_distribution", "reason": "Shows distribution across teaching areas"}}, {{"type": "utterance_timeline", "reason": "Shows questioning patterns over time"}}], lesson_filter: [], area_filter: ["3.3"]
- "I want to analyse 1.1 across the lessons? Give a line graph" → needs_graph: true, graph_types: [{{"type": "utterance_timeline", "reason": "Shows 1.1 patterns over time"}}], lesson_filter: [], area_filter: ["1.1"] (ONLY 1.1, do NOT include 3.4)
</graph_examples>

<time_period_examples>
- "How did I start the lesson?" → class_period: beginning
- "What happened during the middle part?" → class_period: middle
- "How did I wrap up the lesson?" → class_period: end
- "Show me examples of my questioning" → class_period: null (no specific time)
</time_period_examples>
</decision_examples>

"""

    async def _build_message_history(self, conversation_history: List[Dict[str, str]], current_message: str, file_ids: List[str]) -> List:
        """Build proper message history for LangChain with intent analysis context"""
        # Build dynamic system prompt with current lesson information
        dynamic_system_prompt = await self._build_dynamic_system_prompt(file_ids)
        
        messages = [SystemMessage(content=dynamic_system_prompt)]
        
        # Add conversation history (keep last 6 messages for better context)
        for msg in conversation_history[-6:]:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                # Truncate long AI responses — intent routing only needs the gist,
                # not full transcript quotes that bloat context and eat output token budget
                truncated = content[:300] + "..." if len(content) > 300 else content
                messages.append(AIMessage(content=truncated))
        
        # Add current message
        messages.append(HumanMessage(content=current_message))
        return messages

    async def _build_dynamic_system_prompt(self, file_ids: List[str]) -> str:
        """Build dynamic system prompt with current lesson information"""
        # Get lesson info from file_ids
        lesson_info = self._get_lesson_info(file_ids)
        
        # Build the dynamic section with clear mapping instructions
        lesson_section = f"""Available lessons:
{lesson_info}

IMPORTANT: When users reference specific lessons, you MUST include the corresponding file IDs in lesson_filter:
- "lesson 1", "first lesson" → Find the lesson with the earliest date in the filename and use its File ID
- "lesson 2", "second lesson" → Find the lesson with the middle date and use its File ID  
- "lesson 3", "third lesson" → Find the lesson with the latest date and use its File ID
- "all lessons" or no specific lesson mentioned → Use empty array [] to include all lessons

CRITICAL: lesson_filter must contain File IDs (numbers), NOT filenames. 
Example: lesson_filter: [37] (for one lesson) or lesson_filter: [37, 36] (for multiple lessons)

<output_requirements>
ENSURE THAT YOU ALWAYS RESPOND with VALID JSON containing:
- class_period: string ("beginning", "middle", "end", or null)
- agent_to_use: string ("general_assistant" or "rag_assistant")
- transformed_query: string (better query to help with response and RAG) DO NOT MENTION GRAPH OR VISUALIZE
- needs_graph: boolean (true if visualization would help answer the query)
- graph_types: array of graph objects with type and reason (e.g., [{{"type": "teaching_area_distribution", "reason": "Shows distribution across areas"}}, {{"type": "utterance_timeline", "reason": "Shows patterns over time"}}]) or null
- lesson_filter: array of file IDs (e.g., [37, 36]) or empty array for all lessons
- area_filter: array of teaching area codes (e.g., ["3.3", "3.4"]) or empty array for all areas

Note: For single graphs, use graph_types with one object. For multiple graphs, use graph_types with multiple objects. The old graph_type and graph_reason fields are deprecated.
</output_requirements>"""
        
        # Combine base prompt with dynamic section
        return self.system_prompt + "\n\n" + lesson_section

    def _get_graph_descriptions_for_prompt(self) -> str:
        """Get formatted graph descriptions for LLM prompt"""
        descriptions = []
        for key, graph in AVAILABLE_GRAPHS.items():
            lesson_support = "✓" if graph["supports_lesson_filter"] else "✗"
            area_support = "✓" if graph["supports_area_filter"] else "✗"
            descriptions.append(
                f"- {key}: {graph['description']} "
                f"(Lesson Filter: {lesson_support}, Area Filter: {area_support})"
            )
        return "\n".join(descriptions)

    def _get_lesson_info(self, file_ids: List[str]) -> str:
        """Get formatted lesson information with both file IDs and names for LLM selection"""
        if not file_ids:
            print(f"DEBUG: No file_ids provided")
            return "No lessons available"
        
        try:
            print(f"DEBUG: Fetching lesson info for file_ids: {file_ids}")
            # Get file information from database (no await needed for Supabase)
            result = self.supabase.table("files").select("file_id, stored_filename").in_("file_id", file_ids).execute()
            
            print(f"DEBUG: Supabase result: {result}")
            print(f"DEBUG: Result data: {result.data}")
            
            if result.data:
                # Sort data by file_id (ascending) to ensure consistent ordering
                sorted_data = sorted(result.data, key=lambda x: x["file_id"])
                print(f"DEBUG: Sorted data by file_id: {sorted_data}")
                
                # Format lesson info for LLM prompt
                lesson_info_lines = []
                for file_data in sorted_data:
                    file_id = file_data["file_id"]
                    filename = file_data["stored_filename"]
                    lesson_info_lines.append(f"- File ID: {file_id}, Filename: {filename}")
                
                lesson_info = "\n".join(lesson_info_lines)
                print(f"DEBUG: Formatted lesson info: {lesson_info}")
                return lesson_info
            else:
                print(f"DEBUG: No data in result")
                return "No lessons available"
            
        except Exception as e:
            print(f"Error retrieving lesson info: {e}")
            return "No lessons available"

    async def analyze_intent(
        self,
        user_message: str,
        file_ids: List[str],
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user intent and determine routing strategy
        
        Returns:
        - agent_to_use: "general_assistant" or "rag_assistant"
        - class_period: "beginning", "middle", "end", or null
        - transformed_query: Enhanced query for better agent response
        - needs_graph: Boolean indicating if graph is needed
        - graph_types: Array of graph objects with type and reason
        - lesson_filter: Array of lesson references for filtering
        - area_filter: Array of teaching area codes for filtering
        """
        try:
            print(f"DEBUG: Starting analyze_intent for message: {user_message}")
            
            # Build message history for LLM
            print("DEBUG: Building message history...")
            messages = await self._build_message_history(conversation_history or [], user_message, file_ids)
            
            print(f"DEBUG: Messages built, count: {len(messages)}")
            print(f"DEBUG: First message content preview: {messages[0].content[:200]}...")
            
            # Get intent analysis from LLM
            print("DEBUG: Calling LLM...")
            response = await self.llm.ainvoke(messages)
            content = response.content
            print(f"DEBUG: LLM response received, length: {len(content)}")
            print(f"DEBUG: LLM response preview: {content[:500]}...")
            
            # Parse JSON response
            print("DEBUG: Parsing JSON response...")
            try:
                intent_analysis = json.loads(content)
                print(f"DEBUG: JSON parsed successfully: {intent_analysis}")
            except json.JSONDecodeError as json_err:
                print(f"DEBUG: JSON parsing failed: {json_err}")
                # Fallback to basic analysis if JSON parsing fails
                intent_analysis = {
                    "agent_to_use": "general_assistant",
                    "class_period": None,
                    "transformed_query": user_message,
                    "needs_graph": False,
                    "graph_types": [],
                    "lesson_filter": [],
                    "area_filter": []
                }
            
            print("DEBUG: Processing lesson filtering...")
            # Get lesson filter directly from LLM (should contain file IDs)
            lesson_filter = intent_analysis.get("lesson_filter", [])
            print(f"DEBUG: Lesson filter from LLM: {lesson_filter}")
            
            # Validate that lesson_filter contains valid file IDs
            if lesson_filter:
                # Ensure all items in lesson_filter are integers (file IDs)
                validated_filter = []
                for item in lesson_filter:
                    if isinstance(item, int):
                        validated_filter.append(item)
                    elif isinstance(item, str) and item.isdigit():
                        validated_filter.append(int(item))
                    else:
                        print(f"DEBUG: Invalid lesson filter item: {item} (not a file ID)")
                lesson_filter = validated_filter
                print(f"DEBUG: Validated lesson_filter: {lesson_filter}")
            else:
                print("DEBUG: No lesson_filter found in LLM response")
            
            print("DEBUG: Processing area filtering...")
            # Validate and process area filtering
            area_filter = intent_analysis.get("area_filter", [])
            if area_filter:
                # Map natural language to area codes if needed
                area_filter = self._map_natural_language_to_area_codes(area_filter)
            
            print("DEBUG: Updating filters...")
            print(f"DEBUG: Final lesson_filter before update: {lesson_filter}")
            print(f"DEBUG: Final area_filter before update: {area_filter}")
            # Update with processed filters
            intent_analysis["lesson_filter"] = lesson_filter
            intent_analysis["area_filter"] = area_filter

            # Safety net: if the LLM missed graph intent, catch it via keyword scan
            GRAPH_KEYWORDS = ["graph", "chart", "plot", "visuali", "distribution", "trend", "breakdown", "diagram"]
            if not intent_analysis.get("needs_graph"):
                msg_lower = user_message.lower()
                if any(kw in msg_lower for kw in GRAPH_KEYWORDS):
                    print(f"DEBUG: Keyword safety net triggered for message: {user_message}")
                    intent_analysis["needs_graph"] = True
                    intent_analysis["agent_to_use"] = "general_assistant"
                    if not intent_analysis.get("graph_types"):
                        intent_analysis["graph_types"] = [{
                            "type": "teaching_area_distribution",
                            "reason": "User requested graph visualization"
                        }]

            print(f"DEBUG: Final intent_analysis: {intent_analysis}")
            return intent_analysis
            
        except Exception as e:
            import traceback
            print(f"ERROR in analyze_intent: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback to basic analysis
            return {
                "agent_to_use": "general_assistant",
                "class_period": None,
                "transformed_query": user_message,
                "needs_graph": False,
                "graph_types": [],
                "lesson_filter": [],
                "area_filter": []
            }



    def _map_natural_language_to_area_codes(self, area_references: List[str]) -> List[str]:
        """Map natural language area references to area codes"""
        if not area_references:
            print("DEBUG: No area references to map")
            return []
        
        print(f"DEBUG: Mapping area references: {area_references}")
        
        mapped_codes = []
        available_codes = get_available_area_codes()
        
        for ref in area_references:
            print(f"DEBUG: Processing area reference: {ref}")
            
            # Check if it's already an area code
            if ref in available_codes:
                mapped_codes.append(ref)
                print(f"DEBUG: Direct area code match: {ref}")
            else:
                # Try to map natural language
                codes = map_natural_language_to_area_codes(ref)
                print(f"DEBUG: Natural language mapping for '{ref}': {codes}")
                mapped_codes.extend(codes)
        
        final_codes = list(set(mapped_codes))
        print(f"DEBUG: Final mapped area codes: {final_codes}")
        return final_codes
    
# Global instance
intent_analyzer = IntentAnalyzer()