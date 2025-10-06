import os
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
# You can use one model for all, or separate if you want different temperatures, etc.
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
clf_llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# 5) ROUTER (classifier)

router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict multi-class classifier for ANY domain (science, humanities, etc.). "
     "You receive a user TEXT and an optional CONTEXT. Classify into exactly one label:\n"
     "  • 'grammar'  – there is at least one clear grammar/spelling/punctuation error; no conceptual error detected.\n"
     "  • 'concept'  – the TEXT contains a factual/logical claim that is contradicted by the CONTEXT; no grammar errors.\n"
     "  • 'both'     – both the above conditions are true.\n"
     "  • 'none'     – neither condition is met.\n"
     "\n"
     "Definitions:\n"
     "• Grammar error = objective issues in grammar/spelling/punctuation (e.g., subject–verb disagreement, tense, articles, pronouns, prepositions, run-on sentences, missing determiners, obvious typos). "
     "  Do NOT count style, tone, minor awkwardness, or non-native phrasing as errors.\n"
     "• Conceptual error = the TEXT asserts a claim that the CONTEXT explicitly refutes or clearly contradicts. "
     "  If the CONTEXT is empty, irrelevant, or insufficient to judge, you MUST NOT flag a conceptual error.\n"
     "\n"
     "Decision procedure (strict):\n"
     "1) Detect grammar errors ONLY if unambiguous. If unsure, assume no grammar error.\n"
     "2) Detect conceptual errors ONLY if there is a clear contradiction with the CONTEXT. "
     "   Lack of coverage or ambiguity ⇒ NOT a conceptual error.\n"
     "3) Assign label using: both > grammar > concept > none (apply both if both conditions are true).\n"
     "4) Output EXACT JSON with keys: label (one of 'grammar','concept','both','none'), confidence (0..1, two decimals), reasons (array of short strings). "
     "   Reasons should be brief and objective; for conceptual errors, cite a short phrase from CONTEXT that contradicts the claim. "
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
    # d comes from RunnableParallel that we’ll define below
    # shape: {"payload": {"question":..., "context":...}, "route": {...}}
    out = dict(d["payload"])
    out["route"] = d["route"]
    return out

with_route = RunnableParallel({
    "payload": payload,
    "route": route_chain
}) | RunnableLambda(_flatten)

# 6) FIXER CHAINS

# Grammar-only fixer
grammar_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert English editor. Correct ONLY grammar, spelling, punctuation, and clarity. "
     "Do NOT change scientific meaning. Output:\n"
     "1) Corrected text\n2) Bullet list of grammatical fixes."
    ),
    ("human", "Original text:\n{question}")
])
grammar_chain = grammar_prompt | llm | StrOutputParser()

# Concept-only fixer (use the PDF context to correct physics meaning)
concept_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a physics subject-matter expert. Identify and correct conceptual errors using ONLY the provided context. "
     "If the concept is not covered, say you don't know. Output:\n"
     "1) Corrected conceptually accurate version\n2) Bullet list of conceptual issues found (with brief citations to context quotes). "
     "Do NOT do grammar editing unrelated to meaning."
    ),
    ("human", "Text to review:\n{question}\n\nContext:\n{context}")
])
concept_chain = concept_prompt | llm | StrOutputParser()

# BOTH fixer (conceptual + grammar together)
both_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a physics SME and English editor. Fix BOTH conceptual and grammatical errors. "
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
    # Runs the target chain AND, in parallel, grabs router info to print
    return RunnableParallel({
        "result": chain,                                   # the chain's string output
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

# Wrap each chain
grammar_chain_tagged = tag_chain("grammar", grammar_chain)
concept_chain_tagged = tag_chain("concept", concept_chain)
both_chain_tagged    = tag_chain("both",    both_chain)

# Keep your passthrough, but tag it too so you see which branch fired
none_chain_raw = RunnablePassthrough() | RunnableLambda(lambda d: d["question"])
none_chain_tagged = tag_chain("none", none_chain_raw)

# --- Update the router to use the tagged versions ---
router = RunnableBranch(
    (is_label("grammar"), grammar_chain_tagged),
    (is_label("concept"), concept_chain_tagged),
    (is_label("both"),    both_chain_tagged),
    (is_label("none"),    none_chain_tagged),
    # default as final positional arg (must be Runnable)
    RunnableLambda(lambda d: f"[RouterError] Unknown label: {d.get('route')}"),
)

# 8) Compose: classify → branch
# Input to this top-level chain is the raw user text.
router_chain = with_route | router

# -----------------------
# Demo REPL
# -----------------------
print("Router RAG ready. Paste a sentence to classify/fix (Ctrl+C to exit).")
try:
    while True:
        q = input("\nYour text: ").strip()
        if not q:
            continue
        result = router_chain.invoke(q)
        # If you want to inspect the route decision too:
        # route_info = (with_route | RunnableLambda(lambda d: d["route"])).invoke(q)
        # print("ROUTE:", route_info)
        print("\n--- Result ---\n", result)
except KeyboardInterrupt:
    print("\nBye!")
