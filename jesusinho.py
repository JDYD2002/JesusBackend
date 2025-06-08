from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
import tempfile
import asyncio
import httpx
import shelve
from datetime import datetime
from openai import OpenAI
from gtts import gTTS

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

def limpa_resposta(texto, prompt):
    texto_lower = texto.lower().strip()
    prompt_lower = prompt.lower().strip()
    if texto_lower.startswith(prompt_lower):
        return texto[len(prompt):].strip()
    prompt_simples = prompt_lower.replace("responda em português, por favor.\n", "").strip()
    if texto_lower.startswith(prompt_simples):
        return texto[len(prompt_simples):].strip()
    return texto.strip()

# --- OpenAI async ---
async def chat_openai(texto, retries=2):
    prompt = f"Responda em português, por favor:\n{texto}"
    for i in range(retries):
        try:
            resp = await client_openai.chat.acreate(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            resposta = resp.choices[0].message.content.strip()
            return limpa_resposta(resposta, prompt)
        except Exception as e:
            print(f"Erro OpenAI gpt-4o-mini tentativa {i+1}: {e}")
            await asyncio.sleep(2)
    try:
        resp = await client_openai.chat.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        resposta = resp.choices[0].message.content.strip()
        return limpa_resposta(resposta, prompt)
    except Exception as e:
        print(f"Erro fallback OpenAI gpt-3.5-turbo: {e}")
        return ""

# --- Hugging Face async ---
async def chat_hf(texto):
    prompt = f"Responda em português, por favor:\n{texto}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}
    timeout = 10.0

    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in HF_MODELS:
            url = f"https://api-inference.huggingface.co/models/{model}"
            try:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "").strip()
                return data.get("generated_text", "").strip()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                print(f"Erro chat_hf modelo {model}: {e}")
                continue
    return ""

# --- AI21 async ---
async def chat_ai21(texto):
    prompt = f"Responda em português, por favor:\n{texto}"
    headers = {"Authorization": f"Bearer {AI21_API_KEY}", "Content-Type": "application/json"}
    timeout = 10.0

    async with httpx.AsyncClient(timeout=timeout) as client:
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
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                completions = data.get("completions", [])
                if completions:
                    return completions[0].get("data", {}).get("text", "").strip()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                print(f"Erro chat_ai21 modelo {model}: {e}")
                continue
    return ""

# --- Together AI async ---
async def chat_together(texto):
    prompt = f"Responda em português, por favor:\n{texto}"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    timeout = 10.0

    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in TOGETHER_MODELS:
            url = f"https://api.together.xyz/llm/{model}"
            payload = {
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0.7,
            }
            try:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                return data.get("completion", "").strip()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                print(f"Erro chat_together modelo {model}: {e}")
                continue
    return ""

@app.post("/chat")
async def chat_endpoint(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    for func in (chat_openai, chat_hf, chat_ai21, chat_together):
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
    for func in (chat_openai, chat_hf, chat_ai21, chat_together):
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
    return {"mensagem": "API Jesusinho está rodando com fallback inteligente! 🙌"}
