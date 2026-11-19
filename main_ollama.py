from datetime import datetime
from dotenv import load_dotenv
import uuid
import os
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import gradio as gr
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


load_dotenv()

def get_date():
    """ GEt the current day"""
    return datetime.now().strftime('%Y-%m-%d')


llm = ChatOllama(model="qwen3.5:9b")

conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
system_prompt = """
You are a helpful agent and if you are asked what's the date today?
 use def get_date().
"""

agent = create_agent(model=llm,
                     tools=[get_date],
                     system_prompt=system_prompt,
                     checkpointer=checkpointer
                     )

def chat(message, history, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
    config=config,)
    last_response = response["messages"][-1].content
    return last_response


with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    thread_id=gr.State(value=lambda: str(uuid.uuid4()))
    gr.ChatInterface(fn=chat, additional_inputs=[thread_id])

demo.launch()