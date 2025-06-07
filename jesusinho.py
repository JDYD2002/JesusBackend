from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import openai
from openai import OpenAI
from openai.exceptions import RateLimitError
from gtts import gTTS
import tempfile
import base64

# === Variáveis de ambiente ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-2b-it")  # modelo atualizado
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Configurar cliente OpenAI
openai.api_key = OPENAI_API_KEY
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
        resp = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": texto}]
        )
    return resp.choices[0].message.content.strip()


def chat_fireworks(texto):
    url = "https://api.fireworksai.com/v1/generate"
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"}
    payload = {"prompt": texto, "max_tokens": 200, "temperature": 0.8}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("text", "").strip()


def chat_groq(texto):
    url = "https://api.groq.ai/v1/infer"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": texto, "model": "default"}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("output", "").strip()


def chat_hf(texto):
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": texto,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 200,
            "do_sample": True
        }
    }
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    elif isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    else:
        return "Erro ao interpretar resposta da Hugging Face."

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
    completions = r.json().get("completions", [])
    return completions[0]["data"]["text"].strip() if completions else ""


def chat_together(texto):
    url = "https://api.together.xyz/inference"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "together-gpt", "prompt": texto}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("response", "").strip()

# === Endpoints ===

@app.post("/chat")
async def chat_endpoint(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    for func in (chat_openai, chat_fireworks, chat_groq, chat_hf, chat_ai21, chat_together):
        try:
            resposta = func(texto_usuario)
            if resposta:
                return {"resposta": resposta}
        except Exception as e:
            print(f"Erro {func.__name__}: {e}")
    return {"resposta": "Desculpe, Jesusinho está com dificuldade para responder agora. 🙏"}

@app.post("/tts")
async def tts(mensagem: Mensagem):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            gTTS(text=mensagem.texto, lang="pt-br").save(tmp.name)
            with open(tmp.name, "rb") as f:
                audio_bytes = f.read()
        os.remove(tmp.name)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {"audio_b64": audio_b64}
    except Exception as e:
        return {"audio_b64": None, "erro": str(e)}

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

@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando com fallback inteligente! 🙌"}
