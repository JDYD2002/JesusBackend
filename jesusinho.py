import os
import tempfile
import base64
from gtts import gTTS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import httpx

# ======================== CONFIGURAÇÃO FASTAPI ========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== VARIÁVEIS DE AMBIENTE ========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
AI21_API_KEY = os.environ.get("AI21_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

conversa = [
    {"role": "system", "content":
        "Você é Jesus Cristo, o Filho do Deus Vivo. Responda sempre em português brasileiro. Fale com amor, verdade, compaixão e autoridade espiritual, como registrado nos Evangelhos. Suas respostas devem conter versículos bíblicos com referência (como João 3:16), explicar seu significado com profundidade, e sempre apontar para a salvação, graça, arrependimento e o Reino de Deus. Traga consolo, ensino e correção conforme a Bíblia. Nunca contradiga a Palavra de Deus. Fale como o Bom Pastor que guia Suas ovelhas com sabedoria e poder celestial. Fale com unção e reverência, sempre em Português brasileiro. ✝️📖✨"
    }
]

# ======================== MODELO DE MENSAGEM ========================
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

        # Fallbacks
        async def call_openrouter():
            modelos = [
                 "google/gemma-3-27b-it:free",
                "mistralai/devstral-small:free",
                "google/gemini-2.0-flash-exp:free",
                "deepseek/deepseek-chat-v3-0324:free",
                "qwen/qwq-32b:free"
                # Pode adicionar mais modelos aqui, separados por vírgula
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
                        print(f"OpenRouter falhou com {modelo}:", e)

        async def call_huggingface():
            modelos = [
                "google/flan-t5-xl",
                "gpt2",
                "tiiuae/falcon-7b",
                "facebook/blenderbot-400M-distill"
            ]
            prompt = f"Responda em português brasileiro: {texto_usuario}"
            async with httpx.AsyncClient() as cli:
                for modelo in modelos:
                    try:
                        r = await cli.post(
                            f"https://api-inference.huggingface.co/models/{modelo}",
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
                        print(f"HuggingFace falhou com {modelo}:", e)

        async def call_ai21():
            modelos = ["j1-large", "j1-grande", "j1-jumbo"]
            prompt = f"Responda em português brasileiro:\n{texto_usuario}"
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
                        print(f"AI21 falhou com {modelo}:", e)

        # Ordem de fallback
        for func in [call_openrouter, call_huggingface, call_ai21]:
            resultado = await func()
            if resultado:
                conversa.append({"role": "assistant", "content": resultado})
                return resultado

        return "Desculpe, não consegui responder no momento."

# ======================== ROTAS ========================
@app.post("/responder")
async def responder(mensagem: Mensagem):
    resposta = await mensagem.responder_ia(mensagem.texto)
    return {"resposta": resposta}

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
