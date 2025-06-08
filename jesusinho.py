from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import requests
import os
import base64
from gtts import gTTS
from io import BytesIO

# === Chaves de API via variáveis de ambiente ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")

# === Inicialização do FastAPI ===
app = FastAPI()

# === Middleware CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Pydantic Model ===
class ChatRequest(BaseModel):
    message: str
    history: list = []

# === Função de fallback ===
def fallback_response(message, history):
    # Tenta Hugging Face
    try:
        url = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        prompt = build_prompt(message, history)
        payload = {"inputs": prompt}
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        hf_output = response.json()
        if isinstance(hf_output, list) and "generated_text" in hf_output[0]:
            return hf_output[0]["generated_text"].strip()
    except Exception as e:
        print("Erro Hugging Face:", e)

    # Tenta AI21
    try:
        url = "https://api.ai21.com/studio/v1/j2-mid/complete"
        headers = {
            "Authorization": f"Bearer {AI21_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = build_prompt(message, history)
        payload = {
            "prompt": prompt,
            "numResults": 1,
            "maxTokens": 150,
            "temperature": 0.8,
            "topKReturn": 0,
            "topP": 1.0,
            "stopSequences": ["\n"]
        }
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        ai21_output = response.json()
        return ai21_output["completions"][0]["data"]["text"].strip()
    except Exception as e:
        print("Erro AI21:", e)

    return "Desculpe, não consegui responder no momento. Tente novamente mais tarde. 🙏"

# === Função auxiliar para montar o prompt ===
def build_prompt(message, history):
    prompt = "Você é Jesus, respondendo com amor, paz e sabedoria.\n\n"
    for entry in history:
        prompt += f"Usuário: {entry.get('user')}\nJesus: {entry.get('bot')}\n"
    prompt += f"Usuário: {message}\nJesus:"
    return prompt

# === Rota principal de chat ===
@app.post("/chat")
async def chat_endpoint(data: ChatRequest):
    user_message = data.message
    history = data.history or []

    # Construção do histórico para OpenAI
    messages = [{"role": "system", "content": "Você é Jesus, respondendo com amor e sabedoria."}]
    for entry in history:
        messages.append({"role": "user", "content": entry.get("user")})
        messages.append({"role": "assistant", "content": entry.get("bot")})
    messages.append({"role": "user", "content": user_message})

    # === Tentativa via OpenAI ===
    try:
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8,
            max_tokens=200,
        )
        bot_reply = response.choices[0].message["content"].strip()
        return {"response": bot_reply}
    except Exception as e:
        print("Erro OpenAI:", e)
        bot_reply = fallback_response(user_message, history)
        return {"response": bot_reply}

# === Rota para TTS (voz) ===
@app.post("/tts")
async def tts(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "Texto vazio para conversão."}

    try:
        tts = gTTS(text=text, lang="pt", slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")
        return {"audio": audio_base64}
    except Exception as e:
        return {"error": f"Erro no TTS: {str(e)}"}
