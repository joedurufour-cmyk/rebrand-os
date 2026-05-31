import os, json, re
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import anthropic
import pathlib

app = FastAPI(title="REBRAND.OS", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CLAUDE_MODEL = "claude-opus-4-5"
FRONTEND = pathlib.Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/")
async def root():
    if FRONTEND.exists():
        return FileResponse(str(FRONTEND), media_type="text/html")
    return {"ok": True}

@app.get("/health")
@app.get("/api/health")
async def health():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return {"status":"ok","claude_configured": key.startswith("sk-ant")}

@app.get("/api/test")
async def test():
    key = os.getenv("ANTHROPIC_API_KEY","")
    if not key:
        return {"error": "no key"}
    try:
        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=20,
            messages=[{"role":"user","content":"Say OK"}]
        )
        return {"ok": True, "reply": r.content[0].text}
    except Exception as e:
        return {"error": str(e)}

SYSTEM = """Eres REBRAND.OS — Motor de Rebranding Profesional.
REGLAS ABSOLUTAS:
- NUNCA inventar experiencia, cargos, empresas ni métricas
- Marcar claims no validados: ESTIMATED | UNKNOWN
- P_entailment >= 0.90: toda afirmación anclada al CV original
- Separar: Hecho real | Inferencia | Patch narrativo"""

class Query(BaseModel):
    cv_text: str
    job_description: Optional[str] = ""
    target_market: Optional[str] = "remoto"
    mode: str = "pipeline"

class Chat(BaseModel):
    cv_text: str
    job_description: Optional[str] = ""
    message: str
    history: Optional[List[dict]] = []

def llm(prompt: str, system: str = SYSTEM) -> str:
    key = os.getenv("ANTHROPIC_API_KEY","")
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY no configurada")
    client = anthropic.Anthropic(api_key=key)
    r = client.messages.create(
        model=CLAUDE_MODEL,
        system=system,
        messages=[{"role":"user","content":prompt}],
        max_tokens=4000,
        temperature=0.3
    )
    return r.content[0].text

@app.post("/api/v1/pipeline")
async def pipeline(q: Query):
    if not q.cv_text.strip():
        raise HTTPException(400, "cv_text requerido")

    prompts = {
        "pipeline": f"""Analiza este CV contra el puesto. Responde en JSON válido:
CV: {q.cv_text}
PUESTO: {q.job_description or 'No especificado'}
MERCADO: {q.target_market}

JSON exacto:
{{"diagnostico":{{"perfil":"resumen del perfil real","fortalezas":["f1","f2"],"gaps":["g1","g2"]}},"scoring":{{"fit_score":0,"decision":"APPLY|EXPLORE|SKIP","razon":"justificacion breve","keywords_match":["k1"],"keywords_falta":["k2"]}},"patches":[{{"seccion":"SUMMARY","original":"texto original si existe","patch":"version mejorada","tipo":"Hecho real|Inferencia"}}],"next_action":"accion especifica recomendada"}}""",

        "cv_ats": f"""Genera CV versión ATS para este perfil.
CV ORIGINAL: {q.cv_text}
PUESTO TARGET: {q.job_description or q.target_market}
REGLAS: sin tablas/columnas, headers estándar, keywords del puesto integradas, marcar [ESTIMATED] donde infiras métricas.
Genera el CV completo en texto plano.""",

        "cv_recruiter": f"""Genera CV versión Recruiter (lectura 6-8 segundos).
CV ORIGINAL: {q.cv_text}
PUESTO: {q.job_description or q.target_market}
Prioriza impacto visual, summary de 3 líneas, bullets de logros concretos. Sin inventar.""",

        "linkedin": f"""Optimiza perfil LinkedIn completo.
CV: {q.cv_text}
ROL TARGET: {q.job_description or q.target_market}
Genera: 1) Headline (120 chars) 2) About (primera persona, hook) 3) Bullets por experiencia 4) Top 10 skills 5) Keywords para recruiters""",

        "freelance": f"""Analiza oportunidades freelance. Responde JSON:
CV: {q.cv_text}
MERCADO: {q.target_market}
{{"plataformas":[{{"nombre":"","fit":"HIGH|MED|LOW","razon":""}}],"servicios":[{{"servicio":"","precio_estimado":"","base":"Hecho real|Inferencia"}}],"propuesta_valor":"","red_flags":[""],"template_propuesta":"","gaps":[""]}}""",

        "score": f"""Calcula fit score CV vs puesto. JSON:
CV: {q.cv_text}
PUESTO: {q.job_description}
{{"fit_score":0,"decision":"APPLY|EXPLORE|SKIP","razon":"2 lineas max","keywords_match":[""],"keywords_falta":[""]}}"""
    }

    prompt = prompts.get(q.mode, prompts["pipeline"])

    try:
        raw = llm(prompt)
        parsed = None
        if q.mode in ["pipeline", "freelance", "score"]:
            try:
                clean = re.sub(r'```(?:json)?\n?','',raw).strip().rstrip('`')
                parsed = json.loads(clean)
            except:
                parsed = None
        return {"status":"PASS","mode":q.mode,"data":{"content":raw,"structured":parsed}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Error Claude: {str(e)}")

@app.post("/api/v1/coach")
async def coach(c: Chat):
    msgs = []
    for h in (c.history or [])[-6:]:
        if h.get("role") in ["user","assistant"]:
            msgs.append(h)
    msgs.append({"role":"user","content": f"CV en contexto:\n{c.cv_text}\n\nPUESTO:\n{c.job_description}\n\nPREGUNTA: {c.message}"})
    try:
        key = os.getenv("ANTHROPIC_API_KEY","")
        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(model=CLAUDE_MODEL, system=SYSTEM, messages=msgs, max_tokens=2000, temperature=0.4)
        return {"status":"PASS","reply":r.content[0].text}
    except Exception as e:
        raise HTTPException(502, str(e))

@app.post("/api/v1/export")
async def export(body: dict):
    content = body.get("content","")
    name = body.get("name","cv")
    return StreamingResponse(iter([content.encode()]), media_type="text/plain",
        headers={"Content-Disposition":f'attachment; filename="{name}_{datetime.utcnow().strftime("%Y%m%d")}.txt"'})
