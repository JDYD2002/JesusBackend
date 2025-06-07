#!/usr/bin/env python3
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from openai import OpenAI
from openai.error import RateLimitError
from gtts import gTTS
import tempfile
import base64

# === Variáveis de ambiente ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")  # Defina no GitHub Secrets
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# === Clientes API ===
client_openai = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Mensagem(BaseModel):
    texto: str

# === Funções de chat ===

def chat_openai(texto):
    try:
        resp = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": texto}]
        )
    except RateLimitError:
        # Fallback para modelo com maior quota
        resp = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": texto}]
        )
    return resp.choices[0].message.content.strip()


def chat_fireworks(texto):
    # Verifique conectividade DNS/Internet para api.fireworksai.com
    url = "https://api.fireworksai.com/v1/generate"
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"}
    payload = {"prompt": texto, "max_tokens": 200, "temperature": 0.8}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("text", "").strip()


def chat_groq(texto):
    # Verifique conectividade DNS/Internet para api.groq.ai
    url = "https://api.groq.ai/v1/infer"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": texto, "model": "default"}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("output", "").strip()


def chat_hf(texto):
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": texto}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    result = r.json()
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").strip()
    elif isinstance(result, dict):
        return (result.get("generated_text") or result.get("text") or "").strip()
    return ""


def chat_ai21(texto):
    url = "https://api.ai21.com/studio/v1/j1-large/complete"
    headers = {"Authorization": f"Bearer {AI21_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "prompt": texto,
        "numResults": 1,
        "maxTokens": 200,
        "temperature": 0.7,
        "topP": 0.9
    }
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("completions", [{}])[0].get("data", {}).get("text", "").strip()


def chat_together(texto):
    # Endpoint de inference correto
    url = "https://api.together.xyz/inference"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "together-gpt", "prompt": texto}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

# === Endpoint chat com fallback ===
@app.post("/chat")
async def chat(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    for func in (chat_openai, chat_fireworks, chat_groq, chat_hf, chat_ai21, chat_together):
        try:
            resposta = func(texto_usuario)
            if resposta:
                return {"resposta": resposta}
        except Exception as e:
            print(f"Erro {func.__name__}: {e}")
    return {"resposta": "Desculpe, Jesusinho está com dificuldade para responder agora. 🙏"}

# === TTS (áudio base64) ===
@app.post("/tts")
async def tts(mensagem: Mensagem):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            gTTS(text=mensagem.texto, lang="pt-br").save(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                audio_bytes = f.read()
        os.remove(tmpfile.name)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {"audio_b64": audio_b64}
    except Exception as e:
        return {"audio_b64": None, "erro": str(e)}

# === Versículo e Oração do Dia ===
@app.get("/versiculo")
async def versiculo():
    try:
        return {"resposta": chat_openai("Me dê um versículo bíblico inspirador para hoje.")}
    except Exception as e:
        print(f"Erro versiculo: {e}")
        return {"resposta": "Erro ao obter versículo. 🙏"}

@app.get("/oracao")
async def oracao():
    try:
        return {"resposta": chat_openai("Escreva uma oração curta e edificante para o dia de hoje.")}
    except Exception as e:
        print(f"Erro oracao: {e}")
        return {"resposta": "Erro ao obter oração. 🙏"}

# === Status ===
@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando com fallback inteligente! 🙌"}
