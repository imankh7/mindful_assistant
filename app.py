from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import graph

app = FastAPI(title="Mindful Assistant")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    intent: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )
    return ChatResponse(answer=result["answer"], intent=result["intent"])
