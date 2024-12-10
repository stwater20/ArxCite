import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import faiss

# 安装 NLTK 并下载 punkt
import nltk
nltk.download('punkt')

# 假设论文元数据
paper_metadata = [
    {'title': 'Deep Learning Optimization', 'authors': ['Alice', 'Bob'], 'abstract': 'This paper explores optimization techniques for deep learning. These methods improve efficiency.'},
    {'title': 'Natural Language Processing', 'authors': ['Charlie'], 'abstract': 'An introduction to NLP using transformer models. This paper highlights key advancements.'},
    {'title': 'Computer Vision Advances', 'authors': ['Diana', 'Eve'], 'abstract': 'A survey of modern computer vision techniques. Focuses on CNN and transformers.'},
    {'title': 'Quantum Computing', 'authors': ['Frank'], 'abstract': 'Quantum computing for beginners. Explains foundational concepts.'},
    {'title': 'Data Science Techniques', 'authors': ['Grace'], 'abstract': 'Exploring the world of data science. Covers statistics, machine learning, and big data.'}
]

# 加载 Sentence-BERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# 生成句子嵌入并构建映射
all_embeddings = []
all_sentences = []
metadata_index = []

for i, paper in enumerate(paper_metadata):
    sentences = sent_tokenize(paper['abstract'])  # 分句
    embeddings = model.encode(sentences, show_progress_bar=False)  # 生成句子嵌入

    all_embeddings.extend(embeddings)  # 扁平化嵌入
    all_sentences.extend(sentences)  # 扁平化句子
    metadata_index.extend([(i, j) for j in range(len(sentences))])  # 记录论文索引和句子索引

# 转换为 numpy 数组
all_embeddings = np.array(all_embeddings).astype('float32')

# 构建 FAISS 索引
dimension = all_embeddings.shape[1]
index_faiss = faiss.IndexFlatL2(dimension)
index_faiss.add(all_embeddings)

print(f"FAISS 索引已构建，包含 {index_faiss.ntotal} 个句子嵌入。")



from flask import Flask, request, jsonify, render_template
import faiss

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    query_embedding = model.encode([query]).astype('float32')  # 查询嵌入

    # 检索相关句子
    k = 5
    distances, indices = index_faiss.search(query_embedding, k)

    # 构造结果
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        paper_id, sentence_id = metadata_index[idx]  # 根据索引找到论文和句子
        paper = paper_metadata[paper_id]
        sentence = all_sentences[idx]  # 从扁平化句子列表中取句子
        results.append({
            'sentence': sentence,
            'title': paper['title'],
            'authors': ', '.join(paper['authors']),
            'abstract': paper['abstract'],
            'distance': round(float(distance), 4)
        })

    return jsonify(results)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

