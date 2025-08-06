import json
from typing import Tuple

import helper
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import StorageContext, load_index_from_storage
import os

from sentence_transformers import SentenceTransformer, models
word_embedding_model = models.Transformer("./bert-medical-llm")
polling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

json_dir = "./data/train/answers"
txt_dir = "./data/train/statements"

import faiss

def build_index(documents, model_path="./my-sbert-medical"):
    print(f"Loading custom model from {model_path}")

    # Use HuggingFaceEmbeddings directly
    langchain_embed_model = HuggingFaceEmbeddings(model_name=model_path)
    embed_model = LangchainEmbedding(langchain_embed_model)

    # Get embedding dimension from an actual embedding
    dummy_vec = langchain_embed_model.embed_query("test")  # returns a list[float]
    dim = len(dummy_vec)
    
    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(dim)

    # Wrap with LlamaIndex FAISS vector store
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        storage_context=storage_context,
    )
    
    return index

#Query Index and Return metadata
def match_statement(index, query, top_k=1):
    retriever = index.as_retriever(similarity_top_k = top_k)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        metadata = node.metadata
        results.append({
            "matched_text": node.text,
            "topic": metadata["topic"],
            "is_true": metadata["is_true"],
            "file_id": metadata["file_id"]
        })

    return results

### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    
    Args:
        statement (str): The medical statement to classify
        
    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    # Naive baseline that always returns True for statement classification
    statement_is_true = 1
    docs = helper.load_documents(json_dir=json_dir,txt_dir = txt_dir )

    print("Building vector index")
    index = t.build_index(docs)
    results = t.match_statement(index, query=statement)

    # Simple topic matching based on keywords in topic names
    #statement_topic = match_topic(statement)
    
    # return statement_is_true, statement_topic
    top_result = results[0]  # or handle empty list safely
    return (top_result["is_true"], top_result["topic"])

def match_topic(statement: str) -> int:
    """
    Simple keyword matching to find the best topic match.
    """
    # Load topics mapping
    with open('data/topics.json', 'r') as f:
        topics = json.load(f)
    
    statement_lower = statement.lower()
    best_topic = 0
    max_matches = 0
    
    for topic_name, topic_id in topics.items():
        # Extract keywords from topic name
        keywords = topic_name.lower().replace('_', ' ').replace('(', '').replace(')', '').split()
        
        # Count keyword matches in statement
        matches = sum(1 for keyword in keywords if keyword in statement_lower)
        
        if matches > max_matches:
            max_matches = matches
            best_topic = topic_id
    
    return best_topic
