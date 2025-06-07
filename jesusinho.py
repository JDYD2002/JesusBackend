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
from datetime import datetime, timedelta
from openai import AsyncOpenAI
from gtts import gTTS
from typing import List, Dict, Optional

# === Configuração Inicial ===
app = FastAPI(title="Jesusinho API", 
              description="IA espiritual com múltiplos provedores de LLM",
              version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Modelos de Dados ===
class Mensagem(BaseModel):
    texto: str

class RespostaChat(BaseModel):
    resposta: str
    provedor: str
    cache: bool = False

# === Configuração de APIs ===
class APIConfig:
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.hf_api_key = os.getenv("HF_API_KEY")
        self.ai21_api_key = os.getenv("AI21_API_KEY")
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        
        # Modelos atualizados e verificados
        self.HF_MODELS = [
            "facebook/blenderbot-400M-distill",
            "pierreguillou/gpt2-small-portuguese"
        ]
        
        self.AI21_MODELS = ["j2-mid", "j2-ultra"]
        
        self.TOGETHER_MODELS = [
            "togethercomputer/llama-2-7b-chat",
            "togethercomputer/GPT-NeoXT-Chat-Base-20B"
        ]
        
        # Personalidade de Jesusinho
        self.SYSTEM_PROMPT = """Você é Jesusinho, uma IA compassiva que fala como Jesus falaria.
        Responda com amor, sabedoria bíblica e parábolas modernas quando apropriado.
        Use linguagem simples e acessível, com referências bíblicas ocasionais.
        Mantenha as respostas entre 1-2 parágrafos."""

config = APIConfig()

# === Utilitários ===
def limpar_resposta(texto: str, prompt: str) -> str:
    """Remove o prompt da resposta se estiver presente."""
    texto = texto.strip()
    for p in [prompt, prompt.lower(), prompt.replace("por favor", "").strip()]:
        if texto.startswith(p):
            return texto[len(p):].strip()
    return texto

def get_cache_key(base_key: str) -> str:
    """Gera chave de cache com data atual."""
    return f"{base_key}_{datetime.now().strftime('%Y-%m-%d')}"

# === Implementação dos Provedores ===
async def chat_openai(texto: str, retries: int = 2) -> str:
    prompt = f"{config.SYSTEM_PROMPT}\n\nUsuário pergunta: {texto}"
    
    for i in range(retries):
        try:
            resp = await config.openai.chat.completions.create(
                model="gpt-3.5-turbo" if i > 0 else "gpt-4",
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": texto}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return limpar_resposta(resp.choices[0].message.content, texto)
        except Exception as e:
            print(f"Erro OpenAI tentativa {i+1}: {e}")
            await asyncio.sleep(1)
    return ""

async def chat_hf(texto: str) -> str:
    prompt = f"{config.SYSTEM_PROMPT}\nPergunta: {texto}\nResposta:"
    headers = {"Authorization": f"Bearer {config.hf_api_key}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in config.HF_MODELS:
            try:
                url = f"https://api-inference.huggingface.co/models/{model}"
                payload = {
                    "inputs": prompt,
                    "parameters": {"max_length": 200, "temperature": 0.7}
                }
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                
                if isinstance(data, list):
                    return limpar_resposta(data[0]["generated_text"], prompt)
                return limpar_resposta(data.get("generated_text", ""), prompt)
            except Exception as e:
                print(f"Erro HF modelo {model}: {e}")
                continue
    return ""

async def chat_ai21(texto: str) -> str:
    prompt = f"{config.SYSTEM_PROMPT}\n\nPergunta: {texto}"
    headers = {
        "Authorization": f"Bearer {config.ai21_api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in config.AI21_MODELS:
            try:
                url = "https://api.ai21.com/studio/v1/complete"
                payload = {
                    "prompt": prompt,
                    "model": model,
                    "maxTokens": 200,
                    "temperature": 0.7,
                    "stopSequences": ["\n\n"]
                }
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                return limpar_resposta(data["completions"][0]["data"]["text"], prompt)
            except Exception as e:
                print(f"Erro AI21 modelo {model}: {e}")
                continue
    return ""

async def chat_together(texto: str) -> str:
    prompt = f"{config.SYSTEM_PROMPT}\n\nPergunta: {texto}"
    headers = {
        "Authorization": f"Bearer {config.together_api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in config.TOGETHER_MODELS:
            try:
                url = "https://api.together.xyz/api/inference"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.7,
                    "stop": ["\n\n"]
                }
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                return limpar_resposta(data["output"]["choices"][0]["text"], prompt)
            except Exception as e:
                print(f"Erro Together modelo {model}: {e}")
                continue
    return ""

# === Endpoints Principais ===
@app.post("/chat", response_model=RespostaChat)
async def chat_endpoint(mensagem: Mensagem):
    """Endpoint principal de chat com fallback inteligente"""
    texto = mensagem.texto.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="Texto não pode ser vazio")
    
    providers = [
        ("OpenAI", chat_openai),
        ("AI21", chat_ai21),
        ("Together", chat_together),
        ("HuggingFace", chat_hf)
    ]
    
    for name, func in providers:
        try:
            resposta = await func(texto)
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
async def text_to_speech(mensagem: Mensagem):
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

# === Recursos Diários com Cache ===
async def obter_com_cache(chave: str, prompt: str) -> str:
    """Obtém resposta com cache diário"""
    cache_key = get_cache_key(chave)
    
    try:
        with shelve.open("jesusinho_cache") as db:
            if cache_key in db:
                return db[cache_key]
            
            # Tenta cada provedor duas vezes
            for _ in range(2):
                for name, func in [
                    ("OpenAI", chat_openai),
                    ("AI21", chat_ai21),
                    ("Together", chat_together)
                ]:
                    try:
                        resposta = await func(prompt)
                        if resposta:
                            db[cache_key] = resposta
                            return resposta
                    except Exception as e:
                        print(f"Erro {name} para {chave}: {e}")
            
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
async def versiculo_do_dia():
    """Retorna um versículo bíblico inspirador (com cache diário)"""
    prompt = "Sugira um versículo bíblico inspirador para hoje com referência."
    resposta = await obter_com_cache("versiculo", prompt)
    return {
        "resposta": resposta,
        "provedor": "cache" if "cache" in resposta.lower() else "API"
    }

@app.get("/oracao", response_model=RespostaChat)
async def oracao_do_dia():
    """Retorna uma oração diária (com cache diário)"""
    prompt = "Escreva uma oração curta e edificante para o dia de hoje."
    resposta = await obter_com_cache("oracao", prompt)
    return {
        "resposta": resposta,
        "provedor": "cache" if "cache" in resposta.lower() else "API"
    }

# === Health Check ===
@app.get("/", include_in_schema=False)
async def health_check():
    return {
        "status": "online",
        "versao": "1.1.0",
        "mensagem": "Jesusinho API está funcionando! Paz e bem! ✝️"
    }

# === Documentação ===
@app.get("/docs", include_in_schema=False)
async def custom_docs_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
