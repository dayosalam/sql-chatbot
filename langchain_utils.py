from langchain_community.utilities import SQLDatabase


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain.memory import ChatMessageHistory
from example import answer_prompt, api_key
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st


@st.cache_resource
def get_chain():
    print("Creating chain")
    db = SQLDatabase.from_uri("sqlite:///Chinook.db", ignore_tables=['Album', 'Artist',  'Genre',
                                                                     'MediaType', 'Playlist', 'PlaylistTrack',])
    # Large model to be used
    llm = GoogleGenerativeAI(
        model="models/text-bison-001", google_api_key=api_key)
    # SQL query chain
    generate_query = create_sql_query_chain(llm, db)
    # To execute the query
    execute_query = QuerySQLDataBaseTool(db=db)
    rephrase_answer = answer_prompt | llm | StrOutputParser()
    chain = (
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        )
        | rephrase_answer
    )

    return chain


def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message['role'] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


def invoke_chain(question, messages):
    chain = get_chain()
    history = create_history(messages)
    response = chain.invoke(
        {"question": question, "messages": history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    return response
