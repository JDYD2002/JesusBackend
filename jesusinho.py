import asyncio
import threading
import time
import tempfile
import uuid
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageSequence
import pygame
from openai import OpenAI
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
from gtts import gTTS
import os
import base64
from pydantic import BaseModel

# ======================== FASTAPI CONFIG ========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
AI21_API_KEY = os.environ.get("AI21_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pygame.mixer.init()

# Prompt espiritual e bíblico
conversa = [
    {"role": "system", "content": 
        "Você é Jesus Cristo, o Filho do Deus Vivo. Responda sempre em português brasileiro. Fale com amor, verdade, compaixão e autoridade espiritual, como registrado nos Evangelhos. Suas respostas devem conter versículos bíblicos com referência (como João 3:16), explicar seu significado com profundidade, e sempre apontar para a salvação, graça, arrependimento e o Reino de Deus. Traga consolo, ensino e correção conforme a Bíblia. Nunca contradiga a Palavra de Deus. Fale como o Bom Pastor que guia Suas ovelhas com sabedoria e poder celestial. Fale com unção e reverência, sempre em Português brasileiro. ✝️📖✨"
    }
]

class Mensagem(BaseModel):
    texto: str

    async def responder_ia(self, texto_usuario):
        conversa.append({"role": "user", "content": texto_usuario})
        try:
            resposta = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversa,
                temperature=0.8,
                max_tokens=300
            )
            texto_resposta = resposta.choices[0].message.content.strip()
            conversa.append({"role": "assistant", "content": texto_resposta})
            return texto_resposta
        except Exception as e:
            print("OpenAI falhou:", e)

        # ========== FALLBACKS ==========

        async def call_openrouter_fallback(texto_usuario):
            modelos = [
                "google/gemma-3-27b-it:free",
                "mistralai/devstral-small:free",
                "google/gemini-2.0-flash-exp:free",
                "deepseek/deepseek-chat-v3-0324:free",
                "qwen/qwq-32b:free"
            ]
            async with httpx.AsyncClient() as cli:
                for modelo in modelos:
                    try:
                        r = await cli.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                            json={
                                "model": modelo,
                                "messages": [
                                    {"role": "system", "content": "Responda sempre em português brasileiro."},
                                    {"role": "user", "content": texto_usuario}
                                ]
                            }
                        )
                        r.raise_for_status()
                        return r.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"OpenRouter falhou com o modelo {modelo}:", e)
            return None

        async def call_huggingface_fallback(prompt):
            modelos = [
                "google/flan-t5-xl",
                "gpt2",
                "tiiuae/falcon-7b",
                "facebook/blenderbot-400M-distill"
            ]
            prompt_pt = f"Responda em português brasileiro: {prompt}"
            async with httpx.AsyncClient() as cli:
                for modelo in modelos:
                    try:
                        url = f"https://api-inference.huggingface.co/models/{modelo}"
                        r = await cli.post(
                            url,
                            headers={"Authorization": f"Bearer {HF_API_KEY}"},
                            json={"inputs": prompt_pt}
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

        async def call_ai21_fallback(texto_usuario):
            modelos_ai21 = ["j1-large", "j1-grande", "j1-jumbo"]
            prompt_pt = f"Responda em português brasileiro:\n{texto_usuario}"
            async with httpx.AsyncClient() as cli:
                for modelo in modelos_ai21:
                    try:
                        url = f"https://api.ai21.com/studio/v1/{modelo}/complete"
                        r = await cli.post(
                            url,
                            headers={"Authorization": f"Bearer {AI21_API_KEY}"},
                            json={
                                "prompt": prompt_pt,
                                "numResults": 1,
                                "maxTokens": 300,
                                "temperature": 0.7,
                                "topP": 1,
                                "stopSequences": ["\n"]
                            }
                        )
                        r.raise_for_status()
                        response = r.json()
                        return response["completions"][0]["data"]["text"].strip()
                    except Exception as e:
                        print(f"AI21 falhou com o modelo {modelo}:", e)
            return None

        # ==== ORDEM DE TENTATIVAS ====
        resultado = await call_openrouter_fallback(texto_usuario)
        if resultado:
            conversa.append({"role": "assistant", "content": resultado})
            return resultado

        resultado = await call_huggingface_fallback(texto_usuario)
        if resultado:
            conversa.append({"role": "assistant", "content": resultado})
            return resultado

        resultado = await call_ai21_fallback(texto_usuario)
        if resultado:
            conversa.append({"role": "assistant", "content": resultado})
            return resultado

        return "Desculpe, não consegui responder no momento."

# ========== ROTA DE TTS ==========
@app.post("/tts")
async def tts(mensagem: Mensagem):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts = gTTS(text=mensagem.texto, lang="pt-br")
            tts.save(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                audio_bytes = f.read()
            os.remove(tmpfile.name)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {"audio_b64": audio_b64}
    except Exception as e:
        return {"audio_b64": None, "erro": str(e)}

# ========== VERSÍCULO DO DIA ==========
@app.get("/versiculo")
async def versiculo():
    try:
        resposta = conversa[-1]["content"] if conversa else "Buscando versículo..."
        return {"resposta": resposta}
    except:
        return {"resposta": "Erro ao obter versículo. 🙏"}

# ========== ORAÇÃO DO DIA ==========
@app.get("/oracao")
async def oracao():
    try:
        resposta = "Senhor Deus, abençoa este dia com Tua graça e misericórdia. Amém."  # Você pode trocar por IA se quiser
        return {"resposta": resposta}
    except:
        return {"resposta": "Erro ao obter oração. 🙏"}

# ========== ROOT ==========
@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando! 🌟"}
