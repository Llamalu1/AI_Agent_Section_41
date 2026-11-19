from datetime import datetime
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import gradio as gr

load_dotenv()

def get_date():
    """ GEt the current day"""
    return datetime.now().strftime('%Y-%m-%d')

llm = ChatOllama(model="qwen3.5:9b")


system_prompt = """
You are a helpful agent and if you are asked what's the dat today?
 use def get_date().
"""

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)
# user_query = input("Enter query:\n")

# print(response["messages"])

def chat(message):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": " Hi There!"}]})
with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    gr.ChatInterface(fn=chat)

demo.launch()