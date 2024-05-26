import chromadb
from sentence_transformers import SentenceTransformer
from logger import logtool

def get_overlapped_chunks(textin, chunksize, overlapsize):  
    return [textin[a:a+chunksize] for a in range(0,len(textin), chunksize-overlapsize)]



chroma_client = chromadb.Client()
collection_name = "RAG_State_Acts_LawSage_Collection"
collection_flag = False

try:
    existing_collection = chroma_client.get_collection(name=collection_name)
    collection_flag = True
except Exception as e:
    logtool.write_log(e, "RAG")
    collection_flag = False


if collection_flag:
    logtool.write_log(f"{collection_name} collection already exists", "RAG")
else:
    logtool.write_log("Initializing RAG setup...", "RAG")
    dataset = open('datasets/RAG-State Acts Legislation of India.txt', encoding='utf-8').read()
    chunks = get_overlapped_chunks(dataset, 10000, 1000)

    logtool.write_log("Loading embedding model", "RAG")
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    chunk_embeddings = embedding_model.encode(chunks)

    logtool.write_log(f"Creating {collection_name}", "RAG")
    collection = chroma_client.create_collection(name=collection_name)
    max_batch_size = 166

    logtool.write_log(f"Adding documents to {collection_name}", "RAG")
    num_batches = (len(chunk_embeddings) + max_batch_size - 1) // max_batch_size

    for i in range(num_batches):
        start_idx = i * max_batch_size
        end_idx = (i + 1) * max_batch_size
        batch_embeddings = chunk_embeddings[start_idx:end_idx]
        batch_chunks = chunks[start_idx:end_idx]
        batch_ids = [str(j) for j in range(start_idx, min(end_idx, len(chunk_embeddings)))]
    
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_chunks,
            ids=batch_ids
        )

    logtool.write_log("Setup complete", "RAG")





def RAG_state_acts_get_context(query):
    results = collection.query(
        query_embeddings=embedding_model.encode([query]).tolist(),  
        n_results=2
    )
    return results['documents']

