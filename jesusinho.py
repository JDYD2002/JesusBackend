from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from openai import OpenAI
from gtts import gTTS
import tempfile
import base64
import time

# === Variáveis de ambiente ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

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

# Modelos para fallback em cada API
HF_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
]

AI21_MODELS = [
    "j1-large",
    "j1-jumbo",
    "j1-grande",
]

TOGETHER_MODELS = [
    "together-gpt",
    "together-gpt-medium",
    "together-gpt-large",
]

def chat_openai(texto, retries=3):
    for i in range(retries):
        try:
            resp = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": texto}]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Erro no OpenAI gpt-4o-mini, tentativa {i+1}/{retries}: {e}")
            time.sleep(5)
    try:
        resp = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": texto}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro fallback OpenAI gpt-3.5-turbo: {e}")
        return ""

def chat_hf(texto):
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": texto}
    for model in HF_MODELS:
        try:
            url = f"https://api-inference.huggingface.co/models/{model}"
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
            return data.get("generated_text", "").strip()
        except Exception as e:
            print(f"Erro chat_hf modelo {model}: {e}")
            continue
    return ""

def chat_ai21(texto):
    headers = {"Authorization": f"Bearer {AI21_API_KEY}", "Content-Type": "application/json"}
    for model in AI21_MODELS:
        try:
            url = f"https://api.ai21.com/studio/v1/{model}/complete"
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
            if completions:
                return completions[0]["data"]["text"].strip()
        except Exception as e:
            print(f"Erro chat_ai21 modelo {model}: {e}")
            continue
    return ""

def chat_together(texto):
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    for model in TOGETHER_MODELS:
        try:
            url = "https://api.together.xyz/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": texto}],
                "max_tokens": 200,
                "temperature": 0.7,
            }
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices")
            if choices and len(choices) > 0:
                return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"Erro chat_together modelo {model}: {e}")
            continue
    return ""

@app.post("/chat")
async def chat_endpoint(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    for func in (chat_openai, chat_hf, chat_ai21, chat_together):
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
