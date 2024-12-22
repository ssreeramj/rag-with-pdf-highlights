from typing import Annotated, Literal, Sequence, TypedDict

import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# Load the project config file
with open("configs.yaml", "r", encoding="utf8") as file:
    project_configs = yaml.safe_load(file)

load_dotenv()

graph_runnable = None
retriever = None


def update_retriever():
    global graph_runnable, retriever

    if "user_input_db_name" in st.session_state:
        user_input = st.session_state["user_input_db_name"]
        VECTORDB_PATH = f"{project_configs['vectordb-folder-path']}/{user_input}"

        print(f"{user_input=} {VECTORDB_PATH=}")

        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.load_local(
            VECTORDB_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("----- VECTORDB LOADED --------")
        retriever = docsearch.as_retriever()
        print("----- RETRIEVER CREATED --------")

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_product_manuals",
        "Search and Retrieve Information From Product Manuals of ski-vehicles and automobiles.",
    )
    # ---------------------------------

    tools = [retriever_tool]

    class AgentState(TypedDict):
        """agent graph state"""

        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
        messages: Annotated[Sequence[BaseMessage], add_messages]

    ### Nodes
    def grade_documents(state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")

        # Data model
        class Grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)

        # LLM with tool and validation
        llm_with_tool = model.with_structured_output(Grade)

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a \
    user question. \n Here is the retrieved document: \n\n {context} \n\n \
    Here is the user question: {question} \n \
    If the document contains keyword(s) or semantic meaning related to the user \
    question, grade it as relevant. \n \
    Give a binary score 'yes' or 'no' score to indicate whether the document is \
    relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})

        score = scored_result.binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

    ### Nodes
    def agent(state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("---CALL AGENT---")
        messages = state["messages"]
        system_message = """
        You are an expert assistant specializing in ski vehicles and automobiles. Your role is to \
        provide detailed, accurate, and helpful answers to user questions by referring to \
        information from product manuals of ski vehicles. If the user query is related to a \
        vehicle, automobile, automobile parts, or any thing related to driving, \
        then use the tool to retrieve documents from the product manuals.
        """
        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
        model = model.bind_tools(tools)
        response = model.invoke([("system", system_message), *messages])
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    def rewrite(state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n \
    Here is the initial question:\n ------- \n{question} \n ------- \n \
    Formulate an improved question: """,
            )
        ]

        # Grader
        model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
        response = model.invoke(msg)
        return {"messages": [response]}

    def generate(state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # Prompt
        prompt = ChatPromptTemplate.from_template("""You are an expert assistant specializing in \
        ski vehicles. Your role is to provide detailed, accurate, and helpful answers \
        to user questions by referring to information from product manuals of ski vehicles. \
        Respond in a clear and professional tone, ensuring your explanations are easy to \
        understand while providing all necessary details. If the user query cannot be fully \
        answered with the given data, just say that you don't know. \
        If the user query is just a greeting or an acknowledgement, respond back in a friendly manner accordingly.\
        \n\nQuestion: {question} \n\nContext: {context}\n\n  Answer:""")

        # LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

    # Define a new graph
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)

    # Compile
    graph_runnable = workflow.compile()


def invoke_our_graph(st_messages, callables):
    """function to invoke graph"""
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph_runnable.invoke(
        {"messages": st_messages}, config={"callbacks": callables}
    )
