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

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

llm = ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY, temperature=0)
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
    Modify the where condition with OR conditions in the example query below so that it extracts relevant entities from the graph database about user based on a given conversation.
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
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini")
    response = llm.invoke(messages)
    return response.content




class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_state: str
    message_counter: int
    summary: str
    message_for_any_tool: str


@tool
def appt_rescheduler_tool():
    "This is the appt_rescheduler_tool"

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
    if(state["message_counter"]<5): return state
    state["message_counter"]=0
    human_messages_reversed = []
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            human_messages_reversed.append(message.content)
        if len(human_messages_reversed) == 5: 
            break
    human_messages = human_messages_reversed[::-1]
    system_message = f"""
    Extract ONLY HEALTH RELATED important information about user, their actions, their preferences, their condition, their medication and its related information in format of entities and relationships
    and TIE EVERY ENTITY TO USER from the following messages by user:
    Messages: "{human_messages}"
    Return the result in JSON format as shown below without anything else so that it can be loaded in dictionary:
    {{
        "entities": [
            {{"name": "User", "type": "Person"}},
            {{"name": "Medication A", "type": "Medication"}},
        ],
        "relationships": [
            {{"from": "User", "to": "Medication A", "relationship": "is taking"}}
        ]
    }}
    """
    messages = [SystemMessage(content=system_message)]
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
    )
    entities_and_relationships = llm.invoke(messages)
    data = json.loads(entities_and_relationships.content)
    kg.store_entities_and_relationships(data['entities'], data['relationships'])
    return state

def appt_rescheduler(state):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        state["messages"] = [SystemMessage(content=system_message)] + state["messages"]
    prompt = f"""Today's date is: 09/27/24 and day is: Friday. Gather information about new date and time user wants to reschedule appointment to, once you have it,
                call the tool with name "change_state_tool" and pass the new time as argument in "%Y-%m-%d %H:%M:%S" format."""
    messages = [SystemMessage(content=prompt)]+state["messages"]
    llm_with_tool = llm.bind_tools([change_state_tool])
    response = llm_with_tool.invoke(messages)
    message_for_next_tool = ""
    if 'tool_calls' in response.additional_kwargs:
        message_for_next_tool = f"""Patient {patient.first_name} {patient.last_name} is requesting an appointment change from {patient.next_appointment} to {extract_state_from_toolcalls(response)}."""
    return {"messages": [response], "current_state": "appt_rescheduler", "message_for_any_tool": message_for_next_tool}


def assistant(state):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        state["messages"] = [SystemMessage(content=system_message)] + state["messages"]
    for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                last_human_message = message.content
                break
    docs = pc.search(last_human_message)
    context = []
    for doc in docs:
        context.append(doc.page_content)
    prompt = f"""You are a health bot assigned to help users with health related queries and give medical advice. Use the following context to assist the user further.
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.
                        Context: {context}"""
    response = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
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
    llm_with_tool = llm.bind_tools([appt_rescheduler_tool, query_knowledge_graph_tool, assistant_tool, end_tool])
    prompt = """You are just an Orchestrator who calls tools, you don't provide any message.
                Rules to call tools
                - "appt_rescheduler_tool": If the user wants to schedule or reschedule their appointment.
                - "query_knowledge_graph_tool": If the user message is anything related to or asking more about user.
                - "assistant_tool": If the user message is related to health or friendly talk.
                - "end_tool": If the user message is off-topic unrelated to health.
                """
    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = llm_with_tool.invoke(messages)
    return {"messages": [response], "current_state": "orchestrator"}

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


def final_state(state):
    messages = state["messages"]
    if len(messages) > 10:
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = llm.invoke(messages)
        delete_messages = []
        for i, m in enumerate(state["messages"][:-2]):
            delete_messages.append(RemoveMessage(id=m.id))
            if isinstance(m, AIMessage) and "tool_calls" in m.additional_kwargs and (i + 1 < len(state["messages"])) and isinstance(state["messages"][i + 1], ToolMessage):
                delete_messages.append(RemoveMessage(id=state["messages"][i + 1].id))
        kg.close()
        return {"summary": response.content, "messages": delete_messages}
    kg.close()
    return state


def router1(state) -> Literal["orchestrator", "appt_rescheduler"]:
    if state["current_state"] == "appt_rescheduler":
        return "appt_rescheduler"
    else:
        return "orchestrator"

def router2(state) -> Literal["add_appt_rescheduler_tool_message", "add_assistant_tool_message", "query_knowledge_graph", "add_end_tool_message"]:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and "tool_calls" in messages[-1].additional_kwargs and messages[-1].tool_calls[0].get("name") == "appt_rescheduler_tool":
        return "add_appt_rescheduler_tool_message"
    elif isinstance(messages[-1], AIMessage) and "tool_calls" in messages[-1].additional_kwargs and messages[-1].tool_calls[0].get("name") == "query_knowledge_graph_tool":
        return "query_knowledge_graph"
    elif isinstance(messages[-1], AIMessage) and "tool_calls" in messages[-1].additional_kwargs and messages[-1].tool_calls[0].get("name") == "end_tool":
        return "add_end_tool_message"
    return "add_assistant_tool_message"

def router3(state) -> Literal["change_state", "final_state"]:
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
def add_appt_rescheduler_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Calling Appointment rescheduler",
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
g.add_node("appt_rescheduler", appt_rescheduler)
g.add_node("query_knowledge_graph", query_knowledge_graph)
g.add_node("final_state", final_state)

g.add_edge(START, "knowledge_extractor")
g.add_conditional_edges("knowledge_extractor", router1)
g.add_conditional_edges("orchestrator", router2)
g.add_conditional_edges("appt_rescheduler", router3)
g.add_edge("add_appt_rescheduler_tool_message", "appt_rescheduler")
g.add_edge("add_assistant_tool_message", "assistant")
g.add_edge("query_knowledge_graph", "assistant")
g.add_edge("change_state", "final_state")
g.add_edge("add_end_tool_message", "final_state")
g.add_edge("assistant", "final_state")
g.add_edge("final_state", END)
def compile_graph():
    return g.compile(checkpointer=memory)

def display_graph(graph):
    display(Image(graph.get_graph().draw_mermaid_png()))










