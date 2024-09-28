import os
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import List, Literal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, RemoveMessage
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import json
from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from .knowledge_graph import KnowledgeGraph
from .pinecone_store import PineconeStore
from patients.models import Patient
from datetime import datetime
from IPython.display import Image, display
from .llm_adapters.llm_manager import LLMManager


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

llm = LLMManager()
kg = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
pc = PineconeStore(api_key=PINECONE_API_KEY, index_name="healthmate")
patient = Patient.objects.get(first_name="John")

def parse_messages(messages):
    formatted_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted_messages.append(f'Human: "{message.content}"')
        elif isinstance(message, AIMessage):
            if message.content:
              formatted_messages.append(f'AI: "{message.content}"')
        elif isinstance(message, ToolMessage):
            formatted_messages.append(f'Tool: "{message.content}"')
    return "\n".join(formatted_messages)

def extract_state_from_toolcalls(message):
    if 'tool_calls' in message.additional_kwargs:
        tool_calls = message.additional_kwargs['tool_calls']
        for tool_call in tool_calls:
            if 'function' in tool_call and 'arguments' in tool_call['function']:
                arguments = tool_call['function']['arguments']
                try:
                    arguments_dict = json.loads(arguments)
                    return arguments_dict.get('state', None)
                except json.JSONDecodeError:
                    print("Error parsing JSON from arguments.")
                    return None
    return None


def generate_cypher_query_with_llm(user_message, entities, relationships):
    system_message_content = f"""
    You are an expert in querying graph databases.
    Based on the available entities related to the user in the Neo4j database:
    Entities: {entities}
    Modify the where condition by only using OR conditions in the example query below so that it extracts relevant entities from the graph database about user based on a given conversation.
    Example query:
    MATCH path1 = (u:Entity {{name: 'User', type: 'Person'}})-[*]->(target:Entity)
    WHERE target.type = 'Event' OR target.name = 'Cold'
    OPTIONAL MATCH path2 = (target)-[*]->(end:Entity)
    RETURN path1, path2

    Conversation: "{user_message}"
    Return just the Cypher query without adding "cypher" or any quotes so that it can be executed by a tool.
    """
    messages = [
        SystemMessage(content=system_message_content)]
    response = llm.generate_response(messages, bind_tools=False)
    return response.content



class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_state: str
    message_counter: int
    summary: str
    message_for_any_tool: str

@tool
def change_request_tool():
    "This is the change_request_tool"

@tool
def appt_rescheduler_tool():
    "This is the appt_rescheduler_tool"

@tool
def treatment_change_tool():
    "This is the treatment_change_tool"

@tool
def assistant_tool():
    "This is the assistant_tool"

@tool
def query_knowledge_graph_tool(state):
    """This is the "query_knowledge_graph_tool"."""

@tool
def end_tool(state):
    """This is the "end_tool"."""

@tool
def change_state_tool(state):
    """This is the "change_state_tool"."""


def knowledge_extractor(state):
    state["message_counter"]=state["message_counter"]+1
    if(state["message_counter"]<3): return state
    state["message_counter"]=0
    human_messages_reversed = []
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            human_messages_reversed.append(message.content)
        if len(human_messages_reversed) == 5: 
            break
    human_messages = human_messages_reversed[::-1]
    system_message = f"""

Extract all health-related information about the user, including their conditions, symptoms, medications, treatment preferences, and lifestyle factors that could impact their health, such as pet ownership, exercise habits, or dietary choices. Ensure that all relevant entities and relationships are captured, even if they do not directly mention health issues, but could be related (e.g., "I have a dog named Martha" could be relevant for pet allergies).

Capture the following types of information:
- Health conditions (e.g., "I have diabetes.")
- Medications and dosages (e.g., "I am taking 20mg of Lisinopril.")
- Symptoms or complaints (e.g., "I've been experiencing knee pain.")
- Treatment protocols or preferences (e.g., "I prefer physiotherapy over medication.")
- Lifestyle factors (e.g., "I have a dog named Martha.")
- Any other health or lifestyle-related facts or observations that could impact health (e.g., "I exercise regularly" or "I'm allergic to peanuts.")

Tie every relevant entity back to the user, and ensure all extracted entities and relationships are formatted correctly.
Input Messages: "{human_messages}"
Return the result in JSON format as shown below without any other text so that it can be loaded in python dictionary,
if none return empty JSON:
{{
    "entities": [
        {{"name": "User", "type": "Person"}},
        {{"name": "Martha", "type": "Pet", "species": "Dog"}},
        {{"name": "Peanut Allergy", "type": "Allergy"}}
    ],
    "relationships": [
        {{"from": "User", "to": "Martha", "relationship": "owns"}},
        {{"from": "User", "to": "Peanut Allergy", "relationship": "has"}}
    ]
}}
    """
    messages = [SystemMessage(content=system_message)]
    entities_and_relationships = llm.generate_response(messages, bind_tools=False)
    try:
        if "entities" not in entities_and_relationships.content:
            data = {"entities": [], "relationships": []}
        else:
            data = json.loads(entities_and_relationships.content)
    except:
        data = {"entities": [], "relationships": []}
    kg.store_entities_and_relationships(data['entities'], data['relationships'])
    return state

def change_request(state):
    prompt = """You are just an Orchestrator who will call just ONE TOOL and, you DO NOT PROIDE ANY MESSAGE.
                Rules to call tools:
                - **"appt_rescheduler_tool"**: Call this tool if the user expresses any intention to schedule, reschedule, or inquire about an appointment. Example triggers: "I want to reschedule my appointment to next Friday," or "Can I change my appointment time?"
                - **"treatment_change_tool"**: Call this tool when the user is requesting changes to their treatment plan, medication regimen, or other medical interventions. Example triggers: "I need to change my medication," or "Can you adjust my treatment plan?"
                """
    llm.bind_tools([appt_rescheduler_tool, treatment_change_tool])
    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = llm.generate_response(messages, bind_tools=True)
    llm.reset_tools()
    return {"messages": [response], "current_state": "change_request"}

def appt_rescheduler(state):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        state["messages"] = [SystemMessage(content=system_message)] + state["messages"]
    today_date = datetime.now().strftime("%Y-%m-%d")
    day_name = datetime.now().strftime("%A") 
    prompt = f"""Today's date is: {today_date} and day is: {day_name}. Gather information about new date and time user wants to reschedule appointment to, once you have it,
                call the tool with name "change_state_tool" and pass the new time as argument in "%Y-%m-%d %H:%M:%S" format."""
    messages = [SystemMessage(content=prompt)]+state["messages"]
    llm.bind_tools([change_state_tool])
    response = llm.generate_response(messages, bind_tools=True)
    llm.reset_tools()
    message_for_next_tool = ""
    if 'tool_calls' in response.additional_kwargs:
        message_for_next_tool = f"""Patient {patient.first_name} {patient.last_name} is requesting an appointment change from {patient.next_appointment} to {extract_state_from_toolcalls(response)}."""
    return {"messages": [response], "current_state": "appt_rescheduler", "message_for_any_tool": message_for_next_tool}


def treatment_change(state):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        state["messages"] = [SystemMessage(content=system_message)] + state["messages"]
    today_date = datetime.now().strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
    day_name = datetime.now().strftime("%A")
    prompt = f"""Today's date is: {today_date} and day is: {day_name}. Gather information about what specific treatment changes the user is requesting. 
    Once you have the necessary details (e.g., medication changes, dosage adjustments, etc.), call the tool with name "change_state_tool" 
    and pass the changes in a structured format like JSON or key-value pairs."""
    messages = [SystemMessage(content=prompt)] + state["messages"]
    llm.bind_tools([change_state_tool])
    response = llm.generate_response(messages, bind_tools=True)
    llm.reset_tools()  
    message_for_next_tool = ""
    if 'tool_calls' in response.additional_kwargs:
        treatment_changes = extract_state_from_toolcalls(response)
        message_for_next_tool = f"""Patient {patient.first_name} {patient.last_name} is requesting the following treatment changes: {treatment_changes}."""
    return {"messages": [response], "current_state": "treatment_change", "message_for_any_tool": message_for_next_tool}


def change_state(state):
    try:
        doctor_name = patient.doctor_name
    except Patient.DoesNotExist:
        doctor_name = "Doctor"
    return {"messages": [ToolMessage(
                    content=state["message_for_any_tool"],
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"]),
                        AIMessage(
                    content=f"""I will convey your request to {doctor_name}."""
                        )], "current_state": "assistant", "message_for_any_tool": ""}


def assistant(state):
    summary = state.get("summary", "")
    patient_context = (
    f"My name is {patient.first_name} {patient.last_name}. "
    f"I was born on {patient.date_of_birth.strftime('%Y-%m-%d')}. "
    f"My phone number is {patient.phone_number}. "
    f"My email address is {patient.email}. "
    f"My medical condition is {patient.medical_condition}. "
    f"I am taking {patient.medication_regimen}. "
    f"My last appointment was on {patient.last_appointment.strftime('%Y-%m-%d %H:%M')}. "
    f"My next appointment is on {patient.next_appointment.strftime('%Y-%m-%d %H:%M')} "
    f"with {patient.doctor_name}."
        )
    system_message = f"Summary of conversation earlier: {summary} \n\n Patient context: {patient_context}"
    state["messages"] = [SystemMessage(content=system_message)] + state["messages"]
    for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                last_human_message = message.content
                break
    docs = pc.search(last_human_message)
    context = []
    for doc in docs:
        context.append(doc.page_content)
    prompt = f"""You are a health bot assigned to help users with health related and lifestyle queries and give medical advice. Use the following context to assist the user further.
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.
                        Context: {context}"""
    response = llm.generate_response([SystemMessage(content=prompt)] + state["messages"], bind_tools=False)
    return {"messages": [response], "current_state": "assistant"}

def query_knowledge_graph(state):
    messages = parse_messages(state["messages"])
    entities, relationships = kg.fetch_entities_and_relationships_for_user("User")
    cypher_query = generate_cypher_query_with_llm(messages, entities, relationships)
    print(f"Generated Cypher Query: {cypher_query}")
    results = kg.execute_cypher_query(cypher_query)
    return {"messages": [
                ToolMessage(
                    content="\n".join(results),
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            ], "current_state": "assistant"}


def orchestrator(state):
    prompt = """You are just an Orchestrator who will just call ONE tool, and you DONOT PROVIDE ANY MESSAGE
                Rules to call tools:
                - "change_request_tool": Call this tool if the user expresses any intention to change their treatment or appointment. Example triggers: "I want to reschedule my appointment to next Friday," or "Can we change my medication?"
                - "query_knowledge_graph_tool": Call this tool when the user's query is seeking specific information about their health conditions, medications, or other stored health-related details. This is typically triggered by questions about the user's own medical history or related entities. Example triggers: "What medication am I currently taking?" or "Tell me more about my condition."
                - "assistant_tool": Use this tool if the user is asking a general health-related question, seeking advice, or engaging in a friendly conversation. This includes scenarios where the user is looking for recommendations, explanations, or guidance on health-related topics that don't require accessing stored data. Example triggers: "What should I do if I have a cold?" or "Can you tell me more about managing stress?"
                - "end_tool": If the user message is off-topic, unrelated, sensitive, or controversial.
topics"
                """
    llm.bind_tools([change_request_tool, query_knowledge_graph_tool, assistant_tool, end_tool])
    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = llm.generate_response(messages, bind_tools=True)
    llm.reset_tools()
    return {"messages": [response], "current_state": "orchestrator"}



def final_state(state):
    messages = state["messages"]
    if len(messages) > 14:
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Including the previous summary, summarize the new messages above, never miss important user information and medical insights:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = llm.generate_response(messages, bind_tools=False)
        delete_messages = []
        for i, m in enumerate(state["messages"][:-2]):
            delete_messages.append(RemoveMessage(id=m.id))
            if isinstance(m, AIMessage) and "tool_calls" in m.additional_kwargs and (i + 1 < len(state["messages"])) and isinstance(state["messages"][i + 1], ToolMessage):
                delete_messages.append(RemoveMessage(id=state["messages"][i + 1].id))
        kg.close()
        return {"summary": state.get("summary", "")+response.content, "messages": delete_messages}
    kg.close()
    return state


def router1(state) -> Literal["orchestrator", "appt_rescheduler", "treatment_change"]:
    if state["current_state"] == "appt_rescheduler":
        return "appt_rescheduler"
    elif state["current_state"] == "treatment_change":
        return "treatment_change"
    else:
        return "orchestrator"

def router2(state) -> Literal["add_change_request_tool_message", "add_assistant_tool_message", "query_knowledge_graph", "add_end_tool_message"]:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and ((len(messages[-1].content) ==0 and messages[-1].tool_calls[0].get("name") == "change_request_tool") or ("change_request_tool" in messages[-1].content)):
        return "add_change_request_tool_message"
    elif isinstance(messages[-1], AIMessage) and ((len(messages[-1].content) ==0 and messages[-1].tool_calls[0].get("name") == "query_knowledge_graph_tool") or ("query_knowledge_graph_tool" in messages[-1].content)):
        return "query_knowledge_graph"
    elif isinstance(messages[-1], AIMessage) and ((len(messages[-1].content) ==0 and messages[-1].tool_calls[0].get("name") == "end_tool") or ("end_tool" in messages[-1].content)):
        return "add_end_tool_message"
    return "add_assistant_tool_message"

def router3(state) -> Literal["appt_rescheduler", "treatment_change"]:
    messages = state["messages"]
    if isinstance(messages[-2], AIMessage) and "tool_calls" in messages[-2].additional_kwargs and messages[-2].tool_calls[0].get("name") == "appt_rescheduler_tool":
        return "appt_rescheduler"
    elif isinstance(messages[-2], AIMessage) and "tool_calls" in messages[-2].additional_kwargs and messages[-2].tool_calls[0].get("name") == "treatment_change_tool":
        return "treatment_change"
    
def router4(state) -> Literal["change_state", "final_state"]:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and "tool_calls" in messages[-1].additional_kwargs and messages[-1].tool_calls[0].get("name") == "change_state_tool":
        return "change_state"
    return "final_state"

def router5(state) -> Literal["change_state", "final_state"]:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and "tool_calls" in messages[-1].additional_kwargs and messages[-1].tool_calls[0].get("name") == "change_state_tool":
        return "change_state"
    return "final_state"


memory = MemorySaver()
g = StateGraph(State)
g.add_node("knowledge_extractor", knowledge_extractor)
g.add_node("orchestrator", orchestrator)
g.add_node("assistant", assistant)
@g.add_node
def add_change_request_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Will check if the request id for appointment rescheduling or treatment change.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }
@g.add_node
def add_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Need to call the right tool as per user request",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }
@g.add_node
def add_assistant_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Calling Assistant",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }
@g.add_node
def add_end_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Getting in final state",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }
g.add_node("change_state", change_state)
g.add_node("change_request", change_request)
g.add_node("appt_rescheduler", appt_rescheduler)
g.add_node("treatment_change", treatment_change)
g.add_node("query_knowledge_graph", query_knowledge_graph)
g.add_node("final_state", final_state)

g.add_edge(START, "knowledge_extractor")
g.add_conditional_edges("knowledge_extractor", router1)
g.add_conditional_edges("orchestrator", router2)
g.add_conditional_edges("add_tool_message", router3)
g.add_conditional_edges("appt_rescheduler", router4)
g.add_conditional_edges("treatment_change", router5)
g.add_edge("add_change_request_tool_message", "change_request")
g.add_edge("change_request", "add_tool_message")
g.add_edge("add_assistant_tool_message", "assistant")
g.add_edge("query_knowledge_graph", "assistant")
g.add_edge("change_state", "final_state")
g.add_edge("add_end_tool_message", "final_state")
g.add_edge("assistant", "final_state")
g.add_edge("final_state", END)

def compile_graph():
    return g.compile(checkpointer=memory)

def save_graph(graph, file_name='graph.png', output_dir='graphs'):
    os.makedirs(output_dir, exist_ok=True)
    graph_image = graph.get_graph().draw_mermaid_png()
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'wb') as f:
        f.write(graph_image)
    print(f"Graph saved at: {file_path}")