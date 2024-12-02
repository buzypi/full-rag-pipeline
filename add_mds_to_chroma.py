from langchain.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb
import hashlib
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions

# Having issues with UnstructuredMarkdownLoader - it is not reading code snippets from the markdown file
loader = DirectoryLoader(
    "/path/to/md/",
    glob="**/*.md",
    loader_cls=TextLoader,
)

documents = loader.load()
print(f"Found {len(documents)} documents!")

chunk_size = 1000
chunk_overlap = 100
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

splits = text_splitter.split_documents(documents)

sources = {}
for split in splits:
    if split.metadata["source"] not in sources:
        sources[split.metadata["source"]] = 0
    sources[split.metadata["source"]] += 1
    split.metadata["chunk_id"] = split.metadata["source"] + str(
        sources[split.metadata["source"]]
    )

embedding_function = embedding_functions.DefaultEmbeddingFunction()

persistent_client = chromadb.HttpClient(host="172.17.0.2", port=8000)
collection = persistent_client.get_or_create_collection(
    "localmds", embedding_function=embedding_function, metadata={"hnsw:space": "cosine"}
)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="localmds",
    embedding_function=embedding_function,
)

for doc in splits:
    print(f"Adding split for {doc.metadata['chunk_id']}")
    m = hashlib.sha256()
    m.update(doc.metadata["chunk_id"].encode("utf-8"))
    collection.upsert(
        ids=[str(m.hexdigest())], metadatas=doc.metadata, documents=doc.page_content
    )

query = "Tell me about torch"
docs = langchain_chroma.similarity_search_with_score(query)
print(docs[0].page_content)
