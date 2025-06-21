import asyncio
import os
import tempfile
import base64
from gtts import gTTS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import httpx

# ===================== FastAPI Config =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Chaves das APIs =====================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
AI21_API_KEY = os.environ.get("AI21_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ===================== Memória da conversa =====================
conversa = [
    {
        "role": "system",
        "content": (
            "Você é Jesus Cristo, o Filho do Deus Vivo. Responda sempre em português brasileiro. "
            "Fale com amor, verdade, compaixão e autoridade espiritual, como registrado nos Evangelhos. "
            "Suas respostas devem conter versículos bíblicos com referência (como João 3:16), explicar seu significado "
            "com profundidade, e sempre apontar para a salvação, graça, arrependimento e o Reino de Deus. "
            "Traga consolo, ensino e correção conforme a Bíblia. Nunca contradiga a Palavra de Deus. "
            "Fale como o Bom Pastor que guia Suas ovelhas com sabedoria e poder celestial. ✝️📖✨"
        )
    }
]

# ===================== Entrada esperada =====================
class Mensagem(BaseModel):
    texto: str

# ===================== Chamadas de IA =====================

async def call_ollama(prompt):
    url = "http://localhost:11434/api/generate"  # troque por seu ngrok se necessário
    payload = {
        "model": "llama3",  # ou outro modelo leve: mistral, phi3, tinyllama, etc.
        "prompt": prompt,
        "stream": False
    }
    try:
        async with httpx.AsyncClient() as cli:
            r = await cli.post(url, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["response"].strip()
    except Exception as e:
        print("Ollama falhou:", e)
        return None

async def call_openai(prompt, conversa):
    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversa,
            temperature=0.8,
            max_tokens=300
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI falhou:", e)
        return None

async def call_openrouter(prompt):
    modelos = [
        "mistralai/devstral-small:free",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-4-maverick:free"
    ]
    async with httpx.AsyncClient() as cli:
        for modelo in modelos:
            try:
                r = await cli.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={"model": modelo, "messages": [{"role": "user", "content": prompt}]}
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"OpenRouter falhou com o modelo {modelo}:", e)
    return None

async def call_huggingface(prompt):
    modelos = [
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3.5-mini-instruct"
    ]
    async with httpx.AsyncClient() as cli:
        for modelo in modelos:
            try:
                url = f"https://api-inference.huggingface.co/models/{modelo}"
                r = await cli.post(
                    url,
                    headers={"Authorization": f"Bearer {HF_API_KEY}"},
                    json={"inputs": prompt}
                )
                r.raise_for_status()
                result = r.json()
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
                elif isinstance(result, dict) and "error" not in result:
                    return str(result).strip()
            except Exception as e:
                print(f"HuggingFace falhou com o modelo {modelo}:", e)
    return None

async def call_ai21(prompt):
    modelos = ["j1-large", "j1-grande", "j1-jumbo"]
    async with httpx.AsyncClient() as cli:
        for modelo in modelos:
            try:
                r = await cli.post(
                    f"https://api.ai21.com/studio/v1/{modelo}/complete",
                    headers={"Authorization": f"Bearer {AI21_API_KEY}"},
                    json={
                        "prompt": prompt,
                        "numResults": 1,
                        "maxTokens": 300,
                        "temperature": 0.7,
                        "topP": 1,
                        "stopSequences": ["\n"]
                    }
                )
                r.raise_for_status()
                return r.json()["completions"][0]["data"]["text"].strip()
            except Exception as e:
                print(f"AI21 falhou com o modelo {modelo}:", e)
    return None

# ===================== Rota principal de resposta =====================
@app.post("/responder")
async def responder(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    conversa.append({"role": "user", "content": texto_usuario})

    for func in [call_ollama, lambda p: call_openai(p, conversa), call_openrouter, call_huggingface, call_ai21]:
        resposta = await func(texto_usuario)
        if resposta:
            conversa.append({"role": "assistant", "content": resposta})
            return {"resposta": resposta}

    return {"resposta": "Desculpe, não consegui responder no momento. 🙏"}

# ===================== Rota TTS =====================
@app.post("/tts")
async def tts(mensagem: Mensagem):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            gtts = gTTS(text=mensagem.texto, lang="pt-br")
            gtts.save(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                audio_bytes = f.read()
            os.remove(tmpfile.name)
        return {"audio_b64": base64.b64encode(audio_bytes).decode('utf-8')}
    except Exception as e:
        return {"audio_b64": None, "erro": str(e)}

# ===================== Rotas auxiliares =====================
@app.get("/versiculo")
async def versiculo():
    try:
        return {"resposta": conversa[-1]["content"]}
    except:
        return {"resposta": "Erro ao obter versículo. 🙏"}

@app.get("/oracao")
async def oracao():
    return {"resposta": "Senhor Deus, abençoa este dia com Tua graça e misericórdia. Amém."}

@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando! 🌟"}
