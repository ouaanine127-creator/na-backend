"""Minimal OpenAI-compatible backend. Keep provider API keys on this server only."""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Never Alone AI Backend")
client = OpenAI(api_key=os.environ.get("AI_API_KEY"), base_url=os.environ.get("AI_BASE_URL") or None)
MODEL = os.environ.get("AI_MODEL", "gpt-4o-mini")

class Message(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    messages: list[Message]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    if not os.environ.get("AI_API_KEY"):
        raise HTTPException(503, "AI_API_KEY is not configured")
    try:
        messages = [{"role": "system", "content": "You are Never Alone, a friendly, honest, calm AI companion. Never claim to be human or to have human feelings."}]
        messages += [m.model_dump() for m in req.messages[-40:]]
        result = client.chat.completions.create(model=MODEL, messages=messages, timeout=40)
        return {"reply": result.choices[0].message.content or "I could not form a response just now."}
    except Exception as exc:
        raise HTTPException(502, "AI provider request failed") from exc
