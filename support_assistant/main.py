from typing import TypedDict

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import chromadb

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

DOCS_PATH = "docs"

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0"
)


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 3. CREATE CHROMA DATABASE
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="zepto_policies"
)


# ============================================================
# 4. LOAD POLICY DOCUMENTS
# ============================================================

import os

documents = []
metadatas = []
ids = []

for filename in sorted(os.listdir(DOCS_PATH)):

    if filename.endswith(".txt"):

        filepath = os.path.join(
            DOCS_PATH,
            filename
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        documents.append(text)

        metadatas.append({
            "source": filename
        })

        ids.append(filename)


# ============================================================
# 5. CREATE EMBEDDINGS AND STORE IN CHROMA
# ============================================================

if documents:

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

print(
    f"Loaded {len(documents)} policy documents."
)


# ============================================================
# 6. LANGGRAPH STATE
# ============================================================

class SupportState(TypedDict):

    query: str
    intent: str
    answer: str
    sources: list
    confidence: float


# ============================================================
# 7. INTENT CLASSIFICATION
# ============================================================

def classify_intent(
    state: SupportState
):

    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "refund",
        "return",
        "membership",
        "track",
        "tracking",
        "cancel",
        "cancellation",
        "damaged",
        "missing",
        "gift card",
        "support",
        "delivery fee",
        "order"
    ]

    is_policy_question = any(
        keyword in query
        for keyword in policy_keywords
    )

    if is_policy_question:

        intent = "policy_question"

    else:

        intent = "general_question"

    return {
        "intent": intent
    }


# ============================================================
# 8. RETRIEVE POLICY INFORMATION
# ============================================================

def retrieve_and_answer(
    state: SupportState
):

    query = state["query"]

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )

    retrieved_documents = results["documents"][0]
    retrieved_sources = results["metadatas"][0]

    context = "\n\n".join(
        retrieved_documents
    )

    answer = (
        "Based on the available Zepto policy documents:\n\n"
        + context
    )

    sources = [
        item["source"]
        for item in retrieved_sources
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": 1.0
    }


# ============================================================
# 9. GENERAL QUESTION
# ============================================================

def direct_answer(
    state: SupportState
):

    query = state["query"]

    answer = (
        "This is a general question and does not "
        "require retrieval from the Zepto policy documents. "
        f"Question received: {query}"
    )

    return {
        "answer": answer,
        "sources": [],
        "confidence": 0.5
    }


# ============================================================
# 10. ROUTING
# ============================================================

def route_question(
    state: SupportState
):

    if state["intent"] == "policy_question":

        return "policy"

    return "general"


# ============================================================
# 11. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(
    SupportState
)

graph.add_node(
    "classify_intent",
    classify_intent
)

graph.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph.add_node(
    "direct_answer",
    direct_answer
)

graph.add_edge(
    START,
    "classify_intent"
)

graph.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "policy": "retrieve_and_answer",
        "general": "direct_answer"
    }
)

graph.add_edge(
    "retrieve_and_answer",
    END
)

graph.add_edge(
    "direct_answer",
    END
)

support_graph = graph.compile()


# ============================================================
# 12. API SCHEMAS
# ============================================================

class AskRequest(BaseModel):

    query: str


class AskResponse(BaseModel):

    answer: str
    sources: list
    confidence: float


# ============================================================
# 13. FASTAPI ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Zepto Support Assistant is running"
    }


@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(
    request: AskRequest
):

    result = support_graph.invoke({

        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0

    })

    return AskResponse(

        answer=result["answer"],

        sources=result["sources"],

        confidence=result["confidence"]

    )