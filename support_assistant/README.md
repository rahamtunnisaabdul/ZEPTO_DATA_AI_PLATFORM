# Zepto Support Assistant

## Overview

The Zepto Support Assistant is a RAG-based customer support application built using:

- Sentence Transformers
- ChromaDB
- LangGraph
- FastAPI
- Pydantic
- Docker

The system handles two types of questions:

1. Policy-related questions
2. General questions

Policy questions are routed to the retrieval pipeline, while general questions are handled directly.

## Architecture

User Query
    ↓
Intent Classification
    ↓
 ┌─────────────────┐
 │                 │
Policy          General
Question        Question
 │                 │
 ↓                 ↓
ChromaDB        Direct Answer
Retrieval
 │
 ↓
Answer
 │
 ↓
FastAPI Response

## Project Structure

```text
support_assistant/
├── docs/
│   ├── doc_01_delivery.txt
│   ├── doc_02_returns_refunds.txt
│   ├── doc_03_membership.txt
│   ├── doc_04_order_tracking.txt
│   ├── doc_05_cancellation.txt
│   ├── doc_06_damaged_missing.txt
│   ├── doc_07_gift_cards.txt
│   └── doc_08_support_hours.txt
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
## API Testing

The support assistant was tested using the FastAPI Swagger interface.
