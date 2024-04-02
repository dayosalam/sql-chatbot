from langchain_openai import OpenAIEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import os
# Getting API key from another environment file
from dotenv import load_dotenv, find_dotenv
load_dotenv('api_keys.env')
# _ = load_dotenv(find_dotenv()) # read

api_key = os.environ['GEMINI_API_KEY']
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key)


example = [
    {
        "input": "Hey, can you tell me what track number  For Those About To Rock (We Salute You) is?",
        "query": "What is the TrackId for Name = 'For Those About To Rock (We Salute You)'?"
    },

    {"input": "Show details of the invoice for track ID 4",
     "query": "SELECT i.InvoiceId, i.CustomerId, c.FirstName, c.LastName, il.UnitPrice, il.Quantity, i.Total FROM Invoice i JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId JOIN Customer c ON i.CustomerId = c.CustomerId WHERE il.TrackId = 4"
     },

    {"input": "I want to find the contact details for customer Leonie Köhler",
        "query": "SELECT * FROM Customer WHERE FirstName = 'Leonie' AND LastName = 'Köhler'"
     },

    {"input": "What is the total amount invoiced to customer Embraer - Empresa Brasileira de Aeronáutica S.A.?",
        "query": "SELECT SUM(Total) FROM Invoice WHERE CustomerId = (SELECT CustomerId FROM Customer WHERE Company = 'Embraer - Empresa Brasileira de Aeronáutica S.A.')"
     },

    {"input": "List all invoices with a total over $4.00",
        "query": "SELECT * FROM Invoice WHERE Total > 4.00"
     },

    {
        "input": "Change Luis Gonçalves' email to luisg@embraer.com.br",
        "query": "UPDATE Customer SET Email = 'luisg@embraer.com.br' WHERE FirstName = 'Luís' AND LastName = 'Gonçalves';"
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)


# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=example,
#     # input_variables=["input","top_k"],
#     input_variables=["input"],
# )


vectorstore = Chroma()
vectorstore.delete_collection()


@st.cache_resource
def selector():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        example,
        embeddings,
        vectorstore,
        # k=2,
        input_keys=["input"],
    )

    return example_selector


few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=selector(),
    input_variables=["input", "top_k"],
)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, 
    answer the user question in a contructive sentence that mentions the question asked.
    Question: {question}
    SQL Query: {query}
    SQL  Result: {result}
    Answer: """
)


# final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system",
#          "You are a SQLite expert. Given an input question, create a symtactically correct SQLite query to run. Unless otherwise specified. \n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
#         few_shot_prompt,
#         ("human", "{input}"),
#     ]
# )
