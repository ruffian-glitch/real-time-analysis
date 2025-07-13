import json
import re
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(SCRIPT_DIR, 'rag_pdf_chunks.json')

try:
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
except FileNotFoundError:
    print(f"Warning: {CHUNKS_FILE} not found. RAG functionality will be disabled.")
    chunks = []

def retrieve_chunks(query, top_k=3):
    """Retrieve relevant chunks from the knowledge base"""
    # Handle case where chunks are not loaded
    if not chunks:
        return []
    
    # Simple keyword matching: count how many query words appear in each chunk
    query_words = set(re.findall(r'\w+', query.lower()))
    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk['text'].lower()))
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk))
    # Sort by score descending
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:top_k]]

def simple_rag_search(query):
    """Search all .txt files in knowledge base for Rag/"""
    knowledge_dir = os.path.join(SCRIPT_DIR, 'knowledge base for Rag')
    if not os.path.exists(knowledge_dir):
        return []
    
    results = []
    for filename in os.listdir(knowledge_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(knowledge_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        results.append({
                            'source': filename,
                            'content': content[:500] + '...' if len(content) > 500 else content
                        })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return results

if __name__ == '__main__':
    # Example usage
    q = input('Enter your question: ')
    results = retrieve_chunks(q)
    for i, chunk in enumerate(results):
        print(f"\nResult {i+1} (Page {chunk['page']} from {chunk['source']}):\n{chunk['text']}") 