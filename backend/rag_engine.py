from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.retrievers. bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore

try:
    LLM_MODEL_INSTANCE = Ollama(model="tinyllama", base_url="http://127.0.0.1:11434", request_timeout=120.0)
    EMBED_MODEL_INSTANCE = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm, Settings.embed_model = LLM_MODEL_INSTANCE, EMBED_MODEL_INSTANCE
except Exception:
    LLM_MODEL_INSTANCE = None

def init_rag_engine(llm_instance):
    if not llm_instance: return None
    try:
        PERSIST_DIR = "./storage"
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=PERSIST_DIR),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=PERSIST_DIR),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=PERSIST_DIR)
        )
        index=load_index_from_storage(storgae_context)
        node = list(storage_context.docstore.docs.value())
        bm25_retiever=BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
        dende_retriever= index.as_retriever(similarity_top_k=5)
        hybrid_retriever = QueryFusioRetriever([dense_retriever, bm25_retriever]'
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False
        )
        rag_query_engine= index.as_query_engine(
            llm=llm_instance,
            retriever=hybrid_retriever,
            streaming=True          
        )
        return rag_query_engine
    except Exception:
        return None

rag_query_engine = init_rag_engine(LLM_MODEL_INSTANCE)
