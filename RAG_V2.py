# RAG_V3.py — router + tagged outputs + RAGAS-on-each-input (reference-free metrics)

import os
import json
import re
from typing import List, Dict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# -----------------------
# Setup
# -----------------------
load_dotenv()  # expects OPENAI_API_KEY in .env

PDF_PATH = "physics.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index (Chroma)
emb = OpenAIEmbeddings(model="text-embedding-3-small")
persist_dir = "./chroma_db"
vs = Chroma.from_documents(splits, emb, collection_name="rag_docs", persist_directory=persist_dir)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 4) LLMs
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
clf_llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# -----------------------
# 5) ROUTER (classifier)
# -----------------------
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict multi-class classifier for ANY domain (science, humanities, etc.). "
     "You receive a user TEXT and an optional CONTEXT. Classify into exactly one label:\n"
     "  • 'grammar'  – there is at least one clear grammar/spelling/punctuation error; no conceptual error detected.\n"
     "  • 'concept'  – the TEXT contains a factual/logical claim that is contradicted by the CONTEXT; no grammar errors.\n"
     "  • 'both'     – both the above conditions are true.\n"
     "  • 'none'     – neither condition is met.\n\n"
     "Definitions:\n"
     "• Grammar error = objective issues in grammar/spelling/punctuation (e.g., subject–verb disagreement, tense, articles, pronouns, prepositions, run-on sentences, missing determiners, obvious typos). "
     "  Do NOT count style, tone, minor awkwardness, or non-native phrasing as errors.\n"
     "• Conceptual error = the TEXT asserts a claim that the CONTEXT explicitly refutes or clearly contradicts. "
     "  If the CONTEXT is empty, irrelevant, or insufficient to judge, you MUST NOT flag a conceptual error.\n\n"
     "Decision procedure (strict):\n"
     "1) Detect grammar errors ONLY if unambiguous. If unsure, assume no grammar error.\n"
     "2) Detect conceptual errors ONLY if there is a clear contradiction with the CONTEXT. Lack of coverage or ambiguity ⇒ NOT a conceptual error.\n"
     "3) Assign label using: both > grammar > concept > none (apply both if both are true).\n"
     "4) Output EXACT JSON with keys: label (one of 'grammar','concept','both','none'), confidence (0..1, two decimals), reasons (array of short strings). "
     "   Reasons should be brief and objective; for conceptual errors, cite a short phrase from CONTEXT that contradicts the claim.\n"
     "5) Do NOT correct or rewrite the text. Do NOT add explanations outside JSON. No preamble, no code fences."
    ),
    ("human", "TEXT:\n{question}\n\nCONTEXT:\n{context}")
])

router_parser = JsonOutputParser()

# Build an input that provides both question + context
payload = RunnableParallel({
    "question": RunnablePassthrough(),                # the raw user text
    "context": retriever | RunnableLambda(format_docs)  # RAG context
})

# Route decision
route_chain = payload | router_prompt | clf_llm | router_parser

# Flatten payload + route into one dict
def _flatten(d):
    out = dict(d["payload"])
    out["route"] = d["route"]
    return out

with_route = RunnableParallel({
    "payload": payload,
    "route": route_chain
}) | RunnableLambda(_flatten)

# -----------------------
# 6) FIXER CHAINS
# -----------------------
grammar_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert English editor. Correct ONLY grammar, spelling, punctuation, and clarity. "
     "Do NOT change scientific/technical meaning. Output:\n"
     "1) Corrected text\n2) Bullet list of grammatical fixes."
    ),
    ("human", "Original text:\n{question}")
])
grammar_chain = grammar_prompt | llm | StrOutputParser()

concept_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a subject-matter expert. Identify and correct conceptual errors using ONLY the provided context. "
     "If the concept is not covered, say you don't know. Output:\n"
     "1) Corrected conceptually accurate version\n2) Bullet list of conceptual issues found (with brief citations to context quotes). "
     "Do NOT do grammar editing unrelated to meaning."
    ),
    ("human", "Text to review:\n{question}\n\nContext:\n{context}")
])
concept_chain = concept_prompt | llm | StrOutputParser()

both_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are both a subject-matter expert and English editor. Fix BOTH conceptual and grammatical errors. "
     "Use ONLY the provided context for concepts; if context is insufficient, note the uncertainty. "
     "Output:\n"
     "1) Final corrected text (conceptually accurate and grammatically clean)\n"
     "2) Conceptual issues fixed (with brief citations/quotes from context)\n"
     "3) Grammar fixes summary."
    ),
    ("human", "Text to review:\n{question}\n\nContext:\n{context}")
])
both_chain = both_prompt | llm | StrOutputParser()

def is_label(label):
    return lambda x: x.get("route", {}).get("label") == label

# returns the original user text unchanged when no issues
none_chain = RunnablePassthrough() | RunnableLambda(lambda d: d["question"])

# --- Add this helper to tag outputs ---
def tag_chain(name, chain):
    return RunnableParallel({
        "result": chain,                                   # chain's string output
        "route": RunnableLambda(lambda d: d["route"]),     # router JSON (label/confidence/reasons)
    }) | RunnableLambda(
        lambda x: (
            f"[USING: {name.upper()}_CHAIN]"
            f"  label={x['route'].get('label')}"
            f"  confidence={x['route'].get('confidence')}\n"
            f"reasons: {', '.join(x['route'].get('reasons', []))}\n\n"
            f"{x['result']}"
        )
    )

# Wrap each chain with a tag header
grammar_chain_tagged = tag_chain("grammar", grammar_chain)
concept_chain_tagged = tag_chain("concept", concept_chain)
both_chain_tagged    = tag_chain("both",    both_chain)
none_chain_raw       = RunnablePassthrough() | RunnableLambda(lambda d: d["question"])
none_chain_tagged    = tag_chain("none", none_chain_raw)

# --- Router using tagged versions ---
router = RunnableBranch(
    (is_label("grammar"), grammar_chain_tagged),
    (is_label("concept"), concept_chain_tagged),
    (is_label("both"),    both_chain_tagged),
    (is_label("none"),    none_chain_tagged),
    # default as final positional arg (must be Runnable)
    RunnableLambda(lambda d: f"[RouterError] Unknown label: {d.get('route')}"),
)

# 8) Compose: classify → branch
router_chain = with_route | router

# =========================================================
# RAGAS: run ON EACH INPUT (reference-free metrics)
# =========================================================
def _strip_tag_header(text: str) -> str:
    """Remove the '[USING: ...]_CHAIN' header so RAGAS sees only the model's response."""
    if text.startswith("[USING:"):
        parts = text.split("\n\n", 1)
        return parts[1] if len(parts) == 2 else re.sub(r"^\[USING:[^\n]+\]\s*", "", text)
    return text

def _ctx_list_for(query: str, k: int = 4) -> List[str]:
    docs = retriever.invoke(query)
    return [d.page_content for d in docs]

# Try to import RAGAS (and handle version differences) once.
_RAGAS_OK = True
try:
    from ragas import evaluate, EvaluationDataset
    try:
        from ragas.metrics import Faithfulness, ResponseRelevancy
        RelevancyMetric = ResponseRelevancy
    except Exception:
        from ragas.metrics import Faithfulness, AnswerRelevancy
        RelevancyMetric = AnswerRelevancy
    # Prefer modern llm_factory to avoid deprecation warnings
    try:
        from ragas.llms.base import llm_factory
        def make_ragas_llm():
            return llm_factory("gpt-4.1")  # uses OPENAI_API_KEY
    except Exception:
        try:
            from ragas.llms import LangchainLLMWrapper
            def make_ragas_llm():
                return LangchainLLMWrapper(llm)
        except Exception:
            def make_ragas_llm():
                return None
except Exception as e:
    _RAGAS_OK = False
    _RAGAS_ERR = str(e)

def ragas_single_input(q: str, routed_output: str):
    """
    Build a one-row RAGAS dataset and print per-input metrics.
    Uses reference-free metrics: Faithfulness + Response Relevancy.
    """
    if not _RAGAS_OK:
        print("\n[RAGAS] Not available:", _RAGAS_ERR)
        print("Install with: pip install -U ragas datasets pandas")
        return

    response = _strip_tag_header(routed_output)
    contexts = _ctx_list_for(q)

    eval_ds = EvaluationDataset.from_list([{
        "user_input": q,
        "retrieved_contexts": contexts,
        "response": response,
    }])

    ragas_llm = make_ragas_llm()
    if ragas_llm is None:
        print("\n[RAGAS] Could not create judge LLM. Check your ragas install.")
        return

    metrics = [Faithfulness(), RelevancyMetric()]
    results = evaluate(dataset=eval_ds, metrics=metrics, llm=ragas_llm, show_progress=False)

    # Print compact per-input scores
    print("\n=== RAGAS (this input) ===")
    try:
        df = results.to_pandas()
        row = df.iloc[0]
        # Only print the known metric cols if present
        for key in ["faithfulness", "response_relevancy"]:
            if key in df.columns:
                val = row[key]
                print(f"{key}: {val:.3f}" if isinstance(val, (int, float)) else f"{key}: {val}")
        # You can also print all available columns:
        # print(row.to_string())
    except Exception:
        # Fallback: print the raw result object
        print(results)

# -----------------------
# Router Evaluation Metrics
# -----------------------
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

class RouterEvaluator:
    def __init__(self):
        self.predictions = []
        self.confidence_scores = []
        self.true_labels = []
        self.label_counts = defaultdict(int)
        self.confusion_history = []
        
    def add_prediction(self, predicted_label, confidence, true_label=None):
        """Record a prediction and optionally its true label."""
        self.predictions.append(predicted_label)
        self.confidence_scores.append(confidence)
        self.label_counts[predicted_label] += 1
        if true_label:
            self.true_labels.append(true_label)
            self.confusion_history.append((true_label, predicted_label))
    
    def get_stats(self):
        """Get current router statistics."""
        stats = {
            "total_predictions": len(self.predictions),
            "label_distribution": dict(self.label_counts),
            "avg_confidence": np.mean(self.confidence_scores) if self.confidence_scores else 0,
            "confidence_std": np.std(self.confidence_scores) if self.confidence_scores else 0,
        }
        
        # Add accuracy metrics if true labels are available
        if self.true_labels:
            stats["accuracy"] = accuracy_score(self.true_labels, self.predictions)
            cm = confusion_matrix(
                self.true_labels, 
                self.predictions, 
                labels=['grammar', 'concept', 'both', 'none']
            )
            stats["confusion_matrix"] = cm.tolist()
        
        return stats
    
    def print_stats(self):
        """Print formatted router evaluation metrics."""
        stats = self.get_stats()
        print("\n=== Router Evaluation Metrics ===")
        print(f"Total predictions: {stats['total_predictions']}")
        print("\nLabel Distribution:")
        for label, count in stats['label_distribution'].items():
            pct = (count / stats['total_predictions']) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        print(f"\nConfidence Scores:")
        print(f"  Mean: {stats['avg_confidence']:.3f}")
        print(f"  Std:  {stats['confidence_std']:.3f}")
        
        if 'accuracy' in stats:
            print(f"\nAccuracy: {stats['accuracy']:.3f}")
            print("\nConfusion Matrix:")
            labels = ['grammar', 'concept', 'both', 'none']
            print("True\\Pred |", " | ".join(f"{l:7}" for l in labels))
            print("-" * 50)
            for i, true_label in enumerate(labels):
                row = stats['confusion_matrix'][i]
                print(f"{true_label:9}|", " | ".join(f"{x:7d}" for x in row))

# Initialize the router evaluator
router_evaluator = RouterEvaluator()

# -----------------------
# Interactive loop
# -----------------------
print("Router RAG ready. Paste a sentence (Ctrl+C to exit). Each input prints RAGAS metrics too.")
print("To evaluate router performance, add true label after text using '||', e.g.:")
print("'This sentense has errors. || grammar'")

try:
    while True:
        q = input("\nYour text: ").strip()
        if not q:
            continue
            
        # Check for true label in input
        true_label = None
        if "||" in q:
            q, true_label = [x.strip() for x in q.split("||")]
            if true_label not in ['grammar', 'concept', 'both', 'none']:
                print(f"Warning: Invalid true label '{true_label}'. Ignoring.")
                true_label = None
        
        # Get router prediction
        result = router_chain.invoke(q)
        print("\n--- Result ---\n", result)
        
        # Extract router label and confidence from result
        route_info = None
        if isinstance(result, str) and "[USING:" in result:
            label_line = result.split('\n')[0]
            if "label=" in label_line and "confidence=" in label_line:
                label = label_line.split("label=")[1].split()[0]
                confidence = float(label_line.split("confidence=")[1].split()[0])
                route_info = (label, confidence)
        
        # Update router metrics if we got valid routing info
        if route_info:
            label, confidence = route_info
            router_evaluator.add_prediction(label, confidence, true_label)
            router_evaluator.print_stats()
        
        # Run RAGAS metrics for this single input
        ragas_single_input(q, result)
        
except KeyboardInterrupt:
    print("\nBye!")
    # Print final router evaluation
    print("\nFinal Router Evaluation:")
    router_evaluator.print_stats()
