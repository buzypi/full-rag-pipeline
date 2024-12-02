from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import LlamaCpp, OpenAI
from langchain.vectorstores import Chroma
import chromadb
from langchain.prompts import PromptTemplate
import os
import sys

embedding_function = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persistent_client = chromadb.HttpClient(host="172.17.0.2", port=8000)
collection = persistent_client.get_collection("localmds")
langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="localmds",
    embedding_function=embedding_function,
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def get_llm(open_source=True):
    if open_source:
        return LlamaCpp(
        model_path="path-to-your-model.gguf",
        temperature=0.3,
        n_ctx=2500,
        n_threads=19,
        max_tokens=3500,
        top_p=0.5,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    else:
        return OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            openai_api_key=os.environ['OPENAI_API_KEY'])

retriever = langchain_chroma.as_retriever(
    search_type="similarity", search_kwargs={"k": 6}
)

question = ' '.join(sys.argv[1:])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_template = PromptTemplate.from_template(
    """
    You are an intelligent, honest and accurate assistant who works as per instructions.
    Given the following context:

    {context}

    Answer the following question only based on the context and don't repeat yourself:

    {question}
    """
)

retrieved_docs = retriever.get_relevant_documents(question)

prompt = prompt_template.format(context=format_docs(retrieved_docs), question=question)
print(prompt)
sys.exit(0)

llm = get_llm(open_source=True)
print(llm(prompt))
