"""
RAG Knowledge Base & Retrieval Engine.
Indexes merchant advertising network policies, gamification mechanics, and FAQs.
Supports TF-IDF Vector Retrieval (fast, robust, self-contained) and Semantic Search.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

class KnowledgeRetriever:
    """RAG Retriever over merchant advertising knowledge base."""

    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or (DATA_DIR / "knowledge_base.json")
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.doc_vectors = None
        self._load_and_index()

    def _load_and_index(self):
        if not self.kb_path.exists():
            return
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        if not self.documents:
            return

        corpus = [f"{d['title']} {d['content']}" for d in self.documents]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve top_k most relevant knowledge documents."""
        if not self.documents or self.doc_vectors is None:
            return []

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.05: # Relevance threshold
                doc_copy = dict(self.documents[idx])
                doc_copy["relevance_score"] = float(scores[idx])
                results.append(doc_copy)

        return results

    def answer_query(self, query: str) -> str:
        """Produces a grounded merchant support answer."""
        results = self.retrieve(query, top_k=2)
        if not results:
            return "I couldn't find a direct policy on that in our knowledge base. Generally, standard discounts are capped at 50%, campaigns run up to 30 days, and all dynamic QR codes have a 24-hour redemption window."

        primary = results[0]
        secondary_snippet = f"\n\n*Related Policy ({results[1]['title']})*: {results[1]['content']}" if len(results) > 1 else ""

        return f"💡 **{primary['title']}**\n\n{primary['content']}{secondary_snippet}"
