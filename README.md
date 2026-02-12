# mock-openai-server

This mock server implements an OpenAI-compatible HTTP API for testing “typewriter” streaming UX without calling a real model: it exposes /v1/chat/completions and /v1/completions (plus /v1/models), accepts a pre-defined mock_response along with tokens_per_second and time_to_first_token controls, tokenizes the response using tiktoken with GPT-4-compatible encoding, and streams decoded tokens over Server-Sent Events at the requested rate (or returns a standard non-streaming completion when stream=false), making it suitable for validating OpenAI Python client streaming behavior in local apps.

# Installation

```commandline
pip install fastapi uvicorn tiktoken
python mock_openai_tiktoken_stream_server.py
# or:
uvicorn mock_openai_tiktoken_stream_server:app --host 0.0.0.0 --port 8000
```

# Testing

## Commandline
```commandline
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d '{
    "model": "mock-gpt4",
    "stream": true,
    "messages": [{"role":"user","content":"ignored"}],
    "mock_response": "This is streamed by GPT-4 tiktoken tokens ✅. Newline:\nDone.",
    "tokens_per_second": 12,
    "time_to_first_token": 0.8,
    "tokenizer_model": "gpt-4",
    "tokens_per_chunk": 1
  }'
```
## Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="test",  # anything
)

stream = client.chat.completions.create(
    model="mock-gpt4",
    messages=[{"role": "user", "content": "ignored"}],
    stream=True,
    extra_body={
        "mock_response": "Hello — this will stream as GPT-4 tokens via tiktoken.\nAnd respects TTFT + TPS.",
        "tokens_per_second": 12,        # tokens/sec
        "time_to_first_token": 0.8,     # seconds
        "tokenizer_model": "gpt-4",     # tiktoken model name
        "tokens_per_chunk": 1,          # keep 1 for true TPS behavior
    },
)

for event in stream:
    delta = event.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="", flush=True)
print()
```
non-streaming reponse

```python
resp = client.chat.completions.create(
    model="mock-gpt4",
    messages=[{"role": "user", "content": "ignored"}],
    extra_body={
        "mock_response": "Non-stream response.",
        "tokenizer_model": "gpt-4",
    },
)
print(resp.choices[0].message.content)
```
