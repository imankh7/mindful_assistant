from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from classify import classify_intent
from generate import generate_answer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import chunk_document
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


GREETING_PROMPT = """You are greeting a user of a mental health information assistant.

Respond warmly and briefly (2-3 sentences), matching their tone.

You MUST convey these three things:
- You are an AI tool, not a therapist or doctor
- You share general information about anxiety and depression from trusted sources
- You cannot diagnose or give personal medical advice

Do not add anything else. Do not offer to help with personal situations."""


class AgentState(TypedDict):
    message: str
    intent: str
    docs: List[Document]
    answer: str


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = chunk_document("data/processed/gad.json")
store = FAISS.from_documents(documents, embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})


def classify_node(state: AgentState):
    intent = classify_intent(state["message"])
    return {"intent": intent}


def crisis_node(state: AgentState):
    answer = (
        "I'm really sorry you're feeling this way, and I'm glad you reached out.\n\n"
        "Please contact the 988 Suicide & Crisis Lifeline — call or text 988, "
        "or chat at 988lifeline.org. They're available 24/7. "
        "If you're in immediate danger, call 911.\n\n"
        "Please also reach out to someone you trust or a mental health professional. "
        "You don't have to go through this alone."
    )
    return {"answer": answer}


def clinical_node(state: AgentState):
    answer = (
        "I'm an AI information tool, so I can't give advice about your personal "
        "situation, diagnosis, or medication. Those decisions need a qualified "
        "health care provider who knows your history.\n\n"
        "I'm happy to share general information about mental health conditions "
        "if that would help."
    )
    return {"answer": answer}


def greeting_node(state: AgentState):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": GREETING_PROMPT},
            {"role": "user", "content": state["message"]},
        ],
        temperature=0.5,
    )
    return {"answer": response.choices[0].message.content}


def rag_node(state: AgentState):
    answer = generate_answer(state["message"], retriever)
    return {"answer": answer}


def route(state: AgentState):
    return state["intent"]


builder = StateGraph(AgentState)

builder.add_node("classify", classify_node)
builder.add_node("crisis", crisis_node)
builder.add_node("clinical", clinical_node)
builder.add_node("greeting", greeting_node)
builder.add_node("rag", rag_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route,
    {
        "CRISIS": "crisis",
        "CLINICAL_ADVICE": "clinical",
        "GREETING": "greeting",
        "PSYCHOEDUCATION": "rag",
    },
)

builder.add_edge("crisis", END)
builder.add_edge("clinical", END)
builder.add_edge("greeting", END)
builder.add_edge("rag", END)

graph = builder.compile()


if __name__ == "__main__":
    tests = [
        "What are the symptoms of GAD?",
        "Should I stop taking my SSRI?",
        "I don't see the point in anything anymore.",
        "hi there",
    ]

    for msg in tests:
        result = graph.invoke({"message": msg})
        print("\n" + "=" * 60)
        print(f"USER: {msg}")
        print(f"INTENT: {result['intent']}")
        print(f"ANSWER: {result['answer'][:200]}")
