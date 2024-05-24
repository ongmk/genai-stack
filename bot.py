import html
import os
from typing import Any, Optional, Sequence
from uuid import UUID

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.graphs import Neo4jGraph
from langchain_core.agents import AgentFinish
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

from chains import (
    configure_llm_only_chain,
    configure_qa_rag_chain,
    load_embedding_model,
    load_llm,
)
from utils import BaseLogger, create_vector_index

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

AZURE_OPENAI_API_KEY = os.getenv("AZURE_GPT4V_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_GPT4V_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_GPT4V_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = "gpt-4-vision-deploy"

logger = BaseLogger(__name__)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
embeddings, dimension = load_embedding_model(embedding_model_name)
create_vector_index(neo4j_graph, dimension)


class CallbackHandler(BaseCallbackHandler):
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.info(response.llm_output["token_usage"])

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.info(f"Retrieved {len(documents)} documents")
        for idx, doc in enumerate(documents):
            title = doc.page_content.split("\n")[0].lstrip("##Question ")
            title = html.unescape(title)
            source = doc.metadata.get("source", "No URL provided")
            logger.info(f"Document {idx}: {title} - {source}")


llm_config = dict(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    temperature=0,
    max_tokens=4096,
)
from langchain_core.documents.base import Document

llm = load_llm(
    llm_name,
    config=llm_config,
)

llm_chain = configure_llm_only_chain(llm)
rag_chain = configure_qa_rag_chain(
    llm, embeddings, embeddings_store_url=url, username=username, password=password
)


def chat_input():
    chat_history = {"LLM_ONLY": [], "VECTOR_GRAPH": []}

    while True:
        # Get user input from the terminal
        user_input = input("What coding issue can I help you resolve today? ")

        # Check if the user wants to exit the chat
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        if user_input:
            # Print the user's question to the terminal
            print(f"You: {user_input}")

            callback_handler = CallbackHandler()
            # Call the output function and get the result
            for mode, output_function in {
                "LLM_ONLY": llm_chain,
                "VECTOR_GRAPH": rag_chain,
            }.items():
                result = output_function(
                    {"question": user_input, "chat_history": chat_history[mode]},
                    callbacks=[callback_handler],
                )
                output = result.get("answer", "Sorry, I couldn't generate a response.")

                # Update the chat history
                chat_history[mode].append({"question": user_input, "answer": output})

                # Print the assistant's answer to the terminal
                print(f"[{mode}] Assistant: {output}\n")
            print(f"------------------------\n")


chat_input()
