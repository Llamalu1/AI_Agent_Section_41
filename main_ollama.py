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

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
def get_date():
    """ Return today's date in YYYY-MM-DD format."""
    return datetime.now().strftime('%Y-%m-%d')

search_tool = TavilySearchResults()


conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
system_prompt = """
You are a helpful, concise assistant.
Tools:
-get_date: use it whenever the user asks about toda's date, or when ou need the
current date to answer( for ecample " this week" or how many days until... ").
-Search tool: use it for current events, recent news, prices, or anything that 
might have changed recently. Do not use it for general knowledge, math or a casual
converstion.

Rules:
- Before searching for something time-sensitive, call get_date first, so your 
searchquery includes the right date.
- Base answers about current events only on the search results, and which sources
they came from. If the results don't answer the question, say so instead of
guessing.
- If you are unsure of lack information,, say so plainly.
- Remember the earlier messages in this conversation and use them.
- Keep answers short unless the user asks for details.

"""
llm = ChatOllama(model="qwen3.5:9b")

agent = create_agent(model=llm,
                     tools=[get_date, search_tool],
                     system_prompt=system_prompt,
                     checkpointer=checkpointer
                     )

def chat(message, history, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
    config=config)
    last_response = response["messages"][-1].content
    return last_response


with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    thread_id=gr.State(value=lambda: str(uuid.uuid4()))
    gr.ChatInterface(fn=chat, additional_inputs=[thread_id])

demo.launch()