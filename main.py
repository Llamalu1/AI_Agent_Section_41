from datetime import datetime
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
def get_date():
    """ GEt the current day"""
    return datetime.now().strftime('%Y-%m-%d')

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


system_prompt = """
You are a helpful agent and if you are asked what's the dat today?
 use def get_date().
"""

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)
user_query = input("Enter query:\n")
response = agent.invoke({"messages": [{"role":"user", "content": user_query}]})
print(response["messages"][-1].content[0]['text'])