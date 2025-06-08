from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
import tempfile
import asyncio
import httpx
import shelve
import random
from datetime import datetime
from openai import AsyncOpenAI
from gtts import gTTS
from typing import Optional

# === Variáveis de ambiente ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

client_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Mensagem(BaseModel):
    texto: str

class RespostaChat(BaseModel):
    resposta: str
    provedor: str

# Modelos atualizados que funcionam
HF_MODELS = [
    "bert-base-multilingual-uncased",  # Modelo gratuito
    "distilbert-base-uncased"         # Modelo gratuito
]

AI21_MODELS = ["j2-ultra"]  # Modelo premium da AI21

TOGETHER_MODELS = [
    "togethercomputer/RedPajama-INCITE-7B-Chat",  # Modelo gratuito
    "togethercomputer/LLaMA-2-7B-32K"             # Modelo gratuito
]

# Personalidade de Jesusinho
SYSTEM_PROMPT = """Você é Jesusinho, uma IA compassiva que fala como Jesus falaria.
Responda com amor, sabedoria bíblica e parábolas modernas quando apropriado.
Use linguagem simples e acessível, com referências bíblicas ocasionais.
Mantenha as respostas entre 1-2 parágrafos."""

def limpa_resposta(texto: str, prompt: str) -> str:
    """Remove o prompt da resposta se estiver presente."""
    texto = texto.strip()
    variations = [
        prompt,
        prompt.lower(),
        prompt.replace("por favor", "").strip(),
        prompt.replace("Responda em português", "").strip()
    ]
    
    for variation in variations:
        if texto.startswith(variation):
            return texto[len(variation):].strip()
    return texto

# --- OpenAI async --- 
async def chat_openai(texto: str, retries: int = 2) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nUsuário pergunta: {texto}"
    
    for i in range(retries):
        try:
            resp = await client_openai.chat.completions.create(
                model="gpt-3.5-turbo" if i > 0 else "gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": texto}
                ],
                temperature=0.7,
                max_tokens=300
            )
            resposta = resp.choices[0].message.content.strip()
            return limpa_resposta(resposta, texto)
        except Exception as e:
            print(f"Erro OpenAI tentativa {i+1}: {e}")
            await asyncio.sleep(1)
    return ""

# --- Hugging Face async ---
async def chat_hf(texto: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nPergunta: {texto}\nResposta:"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in HF_MODELS:
            try:
                url = f"https://api-inference.huggingface.co/pipeline/text-generation/{model}"
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": 200,
                        "temperature": 0.7,
                        "do_sample": True
                    }
                }
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                
                if isinstance(data, list):
                    return limpa_resposta(data[0]["generated_text"], prompt)
                return limpa_resposta(data.get("generated_text", ""), prompt)
            except Exception as e:
                print(f"Erro HF modelo {model}: {e}")
                continue
    return ""

# --- AI21 async ---
async def chat_ai21(texto: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nPergunta: {texto}"
    headers = {
        "Authorization": f"Bearer {AI21_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            url = "https://api.ai21.com/studio/v1/j2-ultra/complete"
            payload = {
                "prompt": prompt,
                "numResults": 1,
                "maxTokens": 200,
                "temperature": 0.7,
                "topP": 1,
                "stopSequences": ["\n\n"]
            }
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return limpa_resposta(data["completions"][0]["data"]["text"], prompt)
        except Exception as e:
            print(f"Erro AI21: {e}")
            return ""

# --- Together AI async ---
async def chat_together(texto: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nPergunta: {texto}"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in TOGETHER_MODELS:
            try:
                url = "https://api.together.xyz/v1/completions"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.7,
                    "stop": ["\n\n", "Usuário:"]
                }
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                return limpa_resposta(data["choices"][0]["text"], prompt)
            except Exception as e:
                print(f"Erro Together modelo {model}: {e}")
                continue
    return ""

@app.post("/chat", response_model=RespostaChat)
async def chat_endpoint(mensagem: Mensagem):
    """Endpoint principal de chat com fallback inteligente"""
    texto_usuario = mensagem.texto.strip()
    if not texto_usuario:
        raise HTTPException(status_code=400, detail="Texto não pode ser vazio")
    
    providers = [
        ("OpenAI", chat_openai),
        ("AI21", chat_ai21),
        ("Together", chat_together),
        ("HuggingFace", chat_hf)
    ]
    
    for name, func in providers:
        try:
            resposta = await func(texto_usuario)
            if resposta and len(resposta) > 10:  # Filtra respostas muito curtas
                return {
                    "resposta": resposta,
                    "provedor": name
                }
        except Exception as e:
            print(f"Erro no provedor {name}: {str(e)[:200]}")
    
    # Fallback local se todas as APIs falharem
    fallback_responses = [
        "Paz esteja com você! Que Deus abençoe seu dia. 🙏",
        "Jesus te ama incondicionalmente! 'Porque Deus amou o mundo de tal maneira...' (João 3:16)",
        "Busque e você encontrará, bata e a porta se abrirá! (Mateus 7:7)"
    ]
    return {
        "resposta": random.choice(fallback_responses),
        "provedor": "local"
    }

@app.post("/tts")
async def tts(mensagem: Mensagem):
    """Converte texto em áudio (base64 MP3)"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts = gTTS(
                text=mensagem.texto,
                lang="pt-br",
                slow=False,
                lang_check=False
            )
            tts.save(tmp.name)
            
            with open(tmp.name, "rb") as f:
                audio_bytes = f.read()
            
            os.unlink(tmp.name)
            return {
                "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
                "tamanho": len(audio_bytes)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_hoje():
    return datetime.now().strftime("%Y-%m-%d")

async def obter_com_cache(chave: str, prompt: str) -> str:
    """Obtém resposta com cache diário"""
    hoje = get_hoje()
    cache_key = f"{chave}_{hoje}"
    
    try:
        with shelve.open("jesusinho_cache") as db:
            if cache_key in db:
                return db[cache_key]
            
            for func in [chat_openai, chat_ai21, chat_together]:
                try:
                    resposta = await func(prompt)
                    if resposta:
                        db[cache_key] = resposta
                        return resposta
                except Exception as e:
                    print(f"Erro ao gerar {chave}: {e}")
            
            # Fallback local
            fallbacks = {
                "versiculo": "'Amai-vos uns aos outros como eu vos amei.' (João 13:34)",
                "oracao": "Senhor, guia nossos corações hoje. Ajuda-nos a amar como Jesus amou. Amém."
            }
            return fallbacks.get(chave, "Deus abençoe seu dia. 🙏")
    
    except Exception as e:
        print(f"Erro no cache: {e}")
        return "Paz e bem! Ore e tente novamente."

@app.get("/versiculo", response_model=RespostaChat)
async def versiculo():
    prompt = "Me dê um versículo bíblico inspirador e diferente para hoje com referência."
    resposta = await obter_com_cache("versiculo", prompt)
    return {
        "resposta": resposta,
        "provedor": "cache" if "cache" in resposta.lower() else "API"
    }

@app.get("/oracao", response_model=RespostaChat)
async def oracao():
    prompt = "Oração curta, edificante e diferente para o dia de hoje."
    resposta = await obter_com_cache("oracao", prompt)
    return {
        "resposta": resposta,
        "provedor": "cache" if "cache" in resposta.lower() else "API"
    }

@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando com fallback inteligente! 🙌"}

@app.get("/api-status")
async def api_status():
    """Verifica o status de todas as APIs integradas"""
    status = {}
    
    # Teste OpenAI
    try:
        await client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "teste"}],
            max_tokens=1
        )
        status["openai"] = "operacional"
    except Exception as e:
        status["openai"] = f"erro: {str(e)[:100]}"
    
    # Teste AI21
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.ai21.com/studio/v1/models",
                headers={"Authorization": f"Bearer {AI21_API_KEY}"},
                timeout=5
            )
            status["ai21"] = "operacional" if r.status_code == 200 else f"erro HTTP {r.status_code}"
    except Exception as e:
        status["ai21"] = f"erro: {str(e)[:100]}"
    
    # Teste Together
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.together.xyz/v1/models",
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
                timeout=5
            )
            status["together"] = "operacional" if r.status_code == 200 else f"erro HTTP {r.status_code}"
    except Exception as e:
        status["together"] = f"erro: {str(e)[:100]}"
    
    # Teste Hugging Face
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api-inference.huggingface.co/models",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                timeout=5
            )
            status["huggingface"] = "operacional" if r.status_code == 200 else f"erro HTTP {r.status_code}"
    except Exception as e:
        status["huggingface"] = f"erro: {str(e)[:100]}"
    
    return status
