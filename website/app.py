import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
import nltk
nltk.download('punkt')


paper_metadata = [
    {'title': 'Deep Learning Optimization', 'authors': ['Alice', 'Bob'], 'abstract': 'This paper explores optimization techniques for deep learning. These methods improve efficiency.'},
    {'title': 'Natural Language Processing', 'authors': ['Charlie'], 'abstract': 'An introduction to NLP using transformer models. This paper highlights key advancements.'},
    {'title': 'Computer Vision Advances', 'authors': ['Diana', 'Eve'], 'abstract': 'A survey of modern computer vision techniques. Focuses on CNN and transformers.'},
    {'title': 'Quantum Computing', 'authors': ['Frank'], 'abstract': 'Quantum computing for beginners. Explains foundational concepts.'},
    {'title': 'Data Science Techniques', 'authors': ['Grace'], 'abstract': 'Exploring the world of data science. Covers statistics, machine learning, and big data.'}
]


model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

all_embeddings = []
all_sentences = []
metadata_index = []
paper_metadata = []


input_file = '../arxiv-metadata-with-embeddings.json'

print("Loading embeddings and metadata...")
upbound = 10000
with open(input_file, 'r') as f:
    for i, line in enumerate(tqdm(f, desc="Processing JSONL")):
        paper = json.loads(line)
        if 'embeddings' in paper and 'sentences' in paper:
            embeddings = np.array(paper['embeddings'], dtype='float32')
            sentences = paper['sentences']
            all_embeddings.extend(embeddings)
            all_sentences.extend(sentences)
            metadata_index.extend([(i, j) for j in range(len(sentences))])
            paper_metadata.append({
                'id': paper.get('id', ''),
                'title': paper.get('title', 'Unknown'),
                'authors': paper.get('authors', []),
                'abstract': paper.get('abstract', ''),
                'doi': paper.get('doi', ''),
                'categories': paper.get('categories', []),
                'journal-ref': paper.get('journal-ref', '')
            })
        if i == upbound:
            break

# print(paper_metadata[0])


all_embeddings = np.array(all_embeddings).astype('float32')


dimension = all_embeddings.shape[1]
index_faiss = faiss.IndexFlatL2(dimension)
index_faiss.add(all_embeddings)

print(f"FAISS 建構完成，包含 {index_faiss.ntotal} 個 embeddings。")



from flask import Flask, request, jsonify, render_template
import faiss

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    query_embedding = model.encode([query]).astype('float32')  

    # 使用 FAISS 進行最近鄰檢索
    k = 5
    distances, indices = index_faiss.search(query_embedding, k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        paper_id, sentence_id = metadata_index[idx]  # 根據索引找到對應的論文和句子
        paper = paper_metadata[paper_id]
        sentence = all_sentences[idx]

        # 生成 APA 格式引用
        authors = ''.join(paper['authors'])
        year = paper['journal-ref'][-4:] if paper.get('journal-ref') else 'Unknown'
        apa_citation = f"{authors} ({year}). {paper['title']}. Retrieved from {paper['doi']}"

        results.append({
            'sentence': sentence,
            'title': paper['title'],
            'authors': authors,
            'abstract': paper['abstract'],
            'relevance_score': round(float(1 / (1 + distance)), 4),  # 距離轉為相似度
            'apa_citation': apa_citation,
            'url': f"https://doi.org/{paper.get('doi', '')}"  # DOI URL
        })

    return jsonify(results)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

