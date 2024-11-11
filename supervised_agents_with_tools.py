#### Langchain Agent with Supervisor Tool
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
model = ChatOpenAI(model="gpt-4o-mini")
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
@ tool
def positive_response_tool(text : str) -> str:
    """Return response in uppercase since the sentiment was positive or neutral"""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You are an helpful assistant"),
            ("human","{text}")
        ]
    )
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"text":text})
    
    return result.upper()
@ tool
def negative_response_tool(text : str) -> str:
    """Return response in lowercase since the sentiment was negative"""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You are an helpful assistant"),
            ("human","{text}")
        ]
    )
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"text":text})
    
    return result.lower()
@ tool
def sentiment_tool(text : str) -> str:
    """Sentiment CLassifier Tool"""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You are an helpful assistant that receives an input message and returns a sentiment score as a float value between 0.0 to 1.0"),
            ("human","{text}")
        ]
    )
    chain = prompt | model
    result = chain.invoke({"text":text})
    
    return result
from langchain.agents import create_tool_calling_agent
tools=[positive_response_tool, negative_response_tool, sentiment_tool]
agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","""You are an helpful assistant that receives an input message and then :
             1. Check for the sentiment of the text received from the user using sentiment_tool always
             2. If the sentiment is positive or neutral, proceed with positive_response_tool else negative_response_tool
             """),
            ("human","{text}"),
            ("placeholder","{agent_scratchpad}")
        ]
    )
agent = create_tool_calling_agent(model, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
query = "Indian football team has been exceptionally good in the recent years"
response = agent_executor.invoke({"text":query})
response['output']
response.get("intermediate_steps")[0]
response.get("intermediate_steps")[1]
