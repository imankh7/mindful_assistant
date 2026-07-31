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
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

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
    messages: Annotated[list, add_messages]
    intent: str
    docs: List[Document]
    answer: str
    risk_score: Annotated[int, lambda old, new: (old or 0) + new]


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = chunk_document("data/processed/gad.json")
store = FAISS.from_documents(documents, embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})


def classify_node(state: AgentState):
    last_message = state["messages"][-1].content
    intent = classify_intent(last_message)
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
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


def clinical_node(state: AgentState):
    answer = (
        "I'm an AI information tool, so I can't give advice about your personal "
        "situation, diagnosis, or medication. Those decisions need a qualified "
        "health care provider who knows your history.\n\n"
        "I'm happy to share general information about mental health conditions "
        "if that would help."
    )
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


def greeting_node(state: AgentState):
    last_message = state["messages"][-1].content
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": GREETING_PROMPT},
            {"role": "user", "content": last_message},
        ],
        temperature=0.5,
    )
    answer = response.choices[0].message.content
    return {
        "answer": response.choices[0].message.content,
        "messages": [AIMessage(content=answer)],
    }


def rag_node(state: AgentState):
    last_message = state["messages"][-1].content
    answer = generate_answer(last_message, retriever)
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


RISK_PROMPT = """Rate the emotional risk level of this message on a scale of 0 to 3.

0: neutral or factual (e.g. "what are anxiety symptoms?")
1: mild distress (e.g. "I've been stressed and tired")
2: moderate distress (e.g. "I feel hopeless and can't cope")
3: severe — self-harm, suicide, or life not worth living

Message: {message}

Respond with only a single digit: 0, 1, 2, or 3."""


def risk_node(state: AgentState):
    last_message = state["messages"][-1].content
    prompt = RISK_PROMPT.format(message=last_message)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip()
    score = 0
    for char in verdict:
        if char in "0123":
            score = int(char)
            break
    return {"risk_score": score}


RISK_THRESHOLD = 4


def route(state: AgentState):
    if state["risk_score"] >= RISK_THRESHOLD:
        return "CRISIS"
    return state["intent"]


builder = StateGraph(AgentState)

builder.add_node("classify", classify_node)
builder.add_node("crisis", crisis_node)
builder.add_node("risk", risk_node)
builder.add_node("clinical", clinical_node)
builder.add_node("greeting", greeting_node)
builder.add_node("rag", rag_node)

builder.add_edge(START, "risk")
builder.add_edge("risk", "classify")

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

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":

    config = {"configurable": {"thread_id": "conversation-1"}}
    tests = [
        "What are the symptoms of GAD?",
        "I've been really stressed lately",
        "work has been overwhelming and I'm exhausted",
        "I just feel worn down and tired of everything",
    ]

    for msg in tests:
        result = graph.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config=config,
        )
        print("\n" + "=" * 60)
        print(f"USER: {msg}")
        print(f"INTENT: {result['intent']}")
        print(f"RISK (cumulative): {result['risk_score']}")
        print(f"ANSWER: {result['answer'][:200]}")

    print("\n" + "=" * 60)
    print("FULL CONVERSATION HISTORY:")
    final_state = graph.get_state(config)
    for m in final_state.values["messages"]:
        print(f"  {type(m).__name__}: {m.content[:60]}")
