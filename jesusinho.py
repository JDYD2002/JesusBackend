from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
import tempfile
import asyncio
import shelve
from datetime import datetime
from gtts import gTTS
from openai import OpenAI
from openai.error import OpenAIError

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
AI21_API_KEY = os.getenv("AI21_API_KEY") or ""
HF_API_KEY = os.getenv("HF_API_KEY") or ""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)

class Mensagem(BaseModel):
    texto: str

HF_MODELS = ["openai-community/gpt2"]
AI21_MODELS = ["j1-large", "j1-jumbo"]

def limpa_resposta(texto, prompt):
    texto_lower = texto.lower().strip()
    prompt_lower = prompt.lower().strip()
    if texto_lower.startswith(prompt_lower):
        return texto[len(prompt):].strip()
    return texto.strip()

async def chat_openai(texto):
    if not OPENAI_API_KEY:
        return ""
    prompt = f"Responda em português, por favor:\n{texto}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            n=1,
        )
        resposta = response.choices[0].message.content.strip()
        return limpa_resposta(resposta, prompt)
    except OpenAIError as e:
        print(f"Erro OpenAI gpt-4o: {e}")
        # Fallback gpt-3.5-turbo
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                n=1,
            )
            resposta = response.choices[0].message.content.strip()
            return limpa_resposta(resposta, prompt)
        except Exception as e2:
            print(f"Erro fallback OpenAI gpt-3.5-turbo: {e2}")
            return ""

# As funções chat_hf e chat_ai21 podem ser as mesmas da versão anterior,
# usando httpx para chamadas async, ou você pode usar requests e rodar em thread.

# Exemplo simples de chat_hf com httpx:
import httpx

async def chat_hf(texto):
    prompt = f"Responda em português, por favor:\n{texto}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    timeout = 20.0
    async with httpx.AsyncClient(timeout=timeout) as client_hf:
        for model in HF_MODELS:
            url = f"https://api-inference.huggingface.co/models/{model}"
            try:
                r = await client_hf.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "").strip()
                return data.get("generated_text", "").strip()
            except Exception as e:
                print(f"Erro chat_hf modelo {model}: {e}")
                continue
    return ""

async def chat_ai21(texto):
    prompt = f"Responda em português, por favor:\n{texto}"
    headers = {"Authorization": f"Bearer {AI21_API_KEY}", "Content-Type": "application/json"}
    timeout = 15.0
    async with httpx.AsyncClient(timeout=timeout) as client_ai21:
        for model in AI21_MODELS:
            url = f"https://api.ai21.com/studio/v1/{model}/complete"
            payload = {
                "prompt": prompt,
                "maxTokens": 300,
                "temperature": 0.7,
                "topP": 1,
                "stopSequences": []
            }
            try:
                r = await client_ai21.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                completions = data.get("completions", [])
                if completions:
                    return completions[0].get("data", {}).get("text", "").strip()
            except Exception as e:
                print(f"Erro chat_ai21 modelo {model}: {e}")
                continue
    return ""

@app.post("/chat")
async def chat_endpoint(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    for func in (chat_openai, chat_hf, chat_ai21):
        try:
            resposta = await func(texto_usuario)
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

def get_hoje():
    return datetime.now().strftime("%Y-%m-%d")

async def obter_com_cache(chave: str, prompt: str):
    hoje = get_hoje()
    cache_key = f"{chave}_{hoje}"
    with shelve.open("cache") as db:
        if cache_key in db:
            return db[cache_key]
    for func in (chat_openai, chat_hf, chat_ai21):
        try:
            resposta = await func(prompt)
            if resposta:
                with shelve.open("cache") as db:
                    db[cache_key] = resposta
                return resposta
        except Exception as e:
            print(f"Erro {func.__name__} ao gerar {chave}: {e}")
    return f"Erro ao obter {chave}. 🙏"

@app.get("/versiculo")
async def versiculo():
    prompt = "Me dê um versículo bíblico inspirador e diferente para hoje."
    resposta = await obter_com_cache("versiculo", prompt)
    return {"resposta": resposta}

@app.get("/oracao")
async def oracao():
    prompt = "Oração curta, edificante e diferente para o dia de hoje."
    resposta = await obter_com_cache("oracao", prompt)
    return {"resposta": resposta}

@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho rodando com fallback e .create() normal! 🙌"}
