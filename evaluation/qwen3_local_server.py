import argparse
import os
import threading
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512


app = FastAPI()
tokenizer = None
model = None
generation_lock = threading.Lock()
MAX_NEW_TOKENS_SAFE_CAP = 65536


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Keep generation single-threaded on one GPU to avoid instability.
    with generation_lock:
        try:
            chat = [{"role": m.role, "content": m.content} for m in req.messages]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            max_new_tokens = min(req.max_tokens, MAX_NEW_TOKENS_SAFE_CAP)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=req.temperature > 0,
                    temperature=max(req.temperature, 1e-6),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = outputs[0][inputs["input_ids"].shape[1]:]
            content = tokenizer.decode(new_ids, skip_special_tokens=True)
            return {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        except RuntimeError as e:
            # CUDA device-side assert can poison the context. Exit process so supervisor restarts cleanly.
            msg = str(e)
            if "CUDA" in msg or "device-side assert" in msg:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                os._exit(1)
            raise HTTPException(status_code=500, detail=f"runtime_error: {msg[:200]}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"server_error: {str(e)[:200]}")


def main():
    global tokenizer, model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
