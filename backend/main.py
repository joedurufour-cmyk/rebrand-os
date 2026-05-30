"""
REBRAND.OS — Backend API Gateway
Pipeline: PERA M1→M7 | FastAPI | OpenAI | WeasyPrint
"""
import os
import json
import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import pathlib
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import openai

app = FastAPI(title="REBRAND.OS API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── KIMI K2.6 CLIENT (OpenAI-compatible) ─────────────────────
KIMI_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_MODEL = "kimi-k2.6"

def get_openai_client():
    api_key = os.getenv("KIMI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="KIMI_API_KEY not configured")
    return openai.OpenAI(api_key=api_key, base_url=KIMI_BASE_URL)

# ── PERA PROMPT INJECTOR (M2:Encoder + M3:Compressor) ─────────
SYSTEM_PROMPT = """You are REBRAND.OS — a deterministic ATS optimization engine.

CORE CONSTRAINTS (INVARIANTS):
- P_entailment ≥ 0.90: EVERY claim in output must be anchored to user-provided CV data
- ¬ hallucinate: NEVER invent roles, companies, dates, skills, or metrics not in input
- ¬ fabricate: If user CV says "3 years", output must say "3 years" — no upgrades
- cos(V_P, V_JD) MAXIMIZE: Maximize semantic alignment between CV and Job Description
- ATS_compliance: Use standard section headers, keyword-dense language, action verbs
- FORMAT: Return structured JSON only when instructed. No markdown in JSON strings.

PERA PIPELINE EXECUTION:
M1 Parse → extract CV entities + JD requirements
M2 Encode → semantic mapping CV↔JD
M3 Compress → identify gaps + opportunities
M4 Structure → ATS-optimized output schema
M5 Validate → entailment check (reject hallucinated claims)
M6 Adapt → format for output target (text/PDF)
M7 Execute → deliver final artifact

FACTUAL ANCHOR RULE:
∀ output claim → ∃ anchor in user CV.
If anchor missing → mark as [SUGGESTED: verify with user] NOT fabricated.
"""

def build_rebranding_prompt(cv_text: str, job_description: str, mode: str = "full") -> str:
    """M1:Parser + M2:Encoder — build dense instruction"""
    if mode == "analyze":
        return f"""TASK: ATS_GAP_ANALYSIS
INPUT_CV:
{cv_text}

TARGET_JD:
{job_description}

OUTPUT_SCHEMA (JSON):
{{
  "ats_score_before": <0-100>,
  "keyword_matches": ["kw1", "kw2"],
  "keyword_gaps": ["missing1", "missing2"],
  "section_analysis": {{
    "summary": "PASS|FAIL|MISSING",
    "experience": "PASS|FAIL|MISSING",
    "skills": "PASS|FAIL|MISSING",
    "education": "PASS|FAIL|MISSING"
  }},
  "top_recommendations": ["rec1", "rec2", "rec3"],
  "estimated_score_after": <0-100>
}}
Return ONLY valid JSON. No preamble."""

    elif mode == "rewrite":
        return f"""TASK: CV_REWRITE_ATS_OPTIMIZED
CONSTRAINT: P_entailment≥0.90 — anchor all claims to CV below
CONSTRAINT: ¬fabricate — no invented experience

INPUT_CV:
{cv_text}

TARGET_JD:
{job_description}

OUTPUT: Full rewritten CV optimized for ATS. Use:
- Standard headers: PROFESSIONAL SUMMARY | EXPERIENCE | SKILLS | EDUCATION
- Action verbs + quantification where CV data supports it
- Keywords from JD naturally integrated
- Plain text, no tables, no columns (ATS-safe)
- Mark any suggested additions as [SUGGESTED]"""

    elif mode == "summary":
        return f"""TASK: PROFESSIONAL_SUMMARY_GENERATION
CONSTRAINT: P_entailment≥0.90

INPUT_CV:
{cv_text}

TARGET_JD:
{job_description}

OUTPUT: 3-4 sentence ATS-optimized professional summary.
Integrate top 3-5 keywords from JD. Anchor all claims to CV."""

    elif mode == "chat":
        return f"""CONTEXT: User is optimizing their CV for a specific role.
CV ON FILE:
{cv_text}

TARGET ROLE:
{job_description}

Respond as REBRAND.OS coach. Be specific, actionable. ¬fabricate."""

    return f"""TASK: FULL_REBRAND
INPUT_CV: {cv_text}
TARGET_JD: {job_description}
Execute full PERA pipeline. Return optimized CV."""


# ── SCHEMAS ───────────────────────────────────────────────────
class AgentQuery(BaseModel):
    cv_text: str
    job_description: str
    mode: str = "analyze"  # analyze | rewrite | summary | chat
    user_message: Optional[str] = None
    conversation_history: Optional[list] = []

class ExportRequest(BaseModel):
    content: str
    title: str = "Optimized CV"
    candidate_name: str = "Candidate"

# ── SERVE FRONTEND ───────────────────────────────────────────
FRONTEND_PATH = pathlib.Path(__file__).parent.parent / "frontend" / "index.html"  # /app/frontend/index.html in Docker

@app.get("/")
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH), media_type="text/html")
    return {"message": "REBRAND.OS API", "docs": "/docs", "health": "/api/health"}

# ── HEALTH ────────────────────────────────────────────────────
@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "REBRAND.OS",
        "version": "1.0.0",
        "kimi_configured": bool(os.getenv("KIMI_API_KEY")),
        "pipeline": "PERA-v1 M1→M7"
    }

# ── MAIN AGENT ENDPOINT ───────────────────────────────────────
@app.post("/api/v1/agent/query")
async def agent_query(query: AgentQuery):
    """
    PERA Pipeline Executor
    M1:Parse → M2:Encode → M3:Compress → M4:Structure → M5:Validate → M6:Adapt → M7:Execute
    """
    if not query.cv_text.strip():
        raise HTTPException(status_code=400, detail="cv_text required")
    if not query.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description required")

    client = get_openai_client()

    # M1→M3: Build compressed prompt
    task_prompt = build_rebranding_prompt(
        query.cv_text,
        query.job_description,
        query.mode
    )

    # M4: Build messages array
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject conversation history (for chat mode)
    if query.conversation_history:
        for turn in query.conversation_history[-6:]:  # last 6 turns max
            if turn.get("role") in ["user", "assistant"]:
                messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current task
    if query.mode == "chat" and query.user_message:
        messages.append({"role": "user", "content": task_prompt + f"\n\nUSER: {query.user_message}"})
    else:
        messages.append({"role": "user", "content": task_prompt})

    # M7: Execute
    try:
        response = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            temperature=0.6,  # Kimi recommended for non-thinking mode
            max_tokens=8000,
            extra_body={"thinking": {"type": "disabled"}}  # disable thinking for fast ATS tasks
        )
        raw_content = response.choices[0].message.content

        # M5: Validate — parse JSON if expected
        parsed_data = None
        if query.mode == "analyze":
            try:
                # Strip markdown fences if present
                clean = re.sub(r'```(?:json)?\n?', '', raw_content).strip().rstrip('`')
                parsed_data = json.loads(clean)
            except json.JSONDecodeError:
                parsed_data = {"raw": raw_content}

        # M6: Structure response per PERA schema
        result = {
            "status": "PASS",
            "mode": query.mode,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "content": raw_content,
                "structured": parsed_data,
            },
            "transducers": [
                {
                    "action": "download",
                    "format": "pdf",
                    "endpoint": "/api/v1/export/pdf",
                    "label": "Export ATS-Safe PDF"
                },
                {
                    "action": "copy",
                    "format": "text",
                    "label": "Copy to clipboard"
                }
            ],
            "meta": {
                "model": KIMI_MODEL,
                "tokens_used": response.usage.total_tokens,
                "pipeline": "PERA-v1"
            }
        }
        return result

    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")


# ── EXPORT (M6:Adapter — ATS-safe plain text) ───────────────
@app.post("/api/v1/export/pdf")
async def export_pdf(req: ExportRequest):
    """Plain text export — ATS safe, no OS dependencies"""
    try:
        txt_bytes = req.content.encode("utf-8")
        filename = f"cv_{req.candidate_name.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}.txt"
        return StreamingResponse(
            iter([txt_bytes]),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


def _text_to_html(text: str) -> str:
    """Convert plain text CV to ATS-safe HTML structure"""
    lines = text.strip().split('\n')
    html_parts = []
    in_list = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            continue

        # Detect section headers (ALL CAPS or common patterns)
        if re.match(r'^(PROFESSIONAL SUMMARY|EXPERIENCE|SKILLS|EDUCATION|CERTIFICATIONS|PROJECTS|SUMMARY|WORK EXPERIENCE|EMPLOYMENT)', line, re.I):
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append(f'<h2>{line}</h2>')

        # Bullet points
        elif line.startswith(('•', '-', '*', '·')):
            if not in_list:
                html_parts.append('<ul>')
                in_list = True
            html_parts.append(f'<li>{line[1:].strip()}</li>')

        # Job title / company lines (has | or similar separator)
        elif '|' in line or re.match(r'.+\d{4}', line):
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append(f'<h3>{line}</h3>')

        else:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append(f'<p>{line}</p>')

    if in_list:
        html_parts.append('</ul>')

    return '\n'.join(html_parts)


# ── KEYWORD EXTRACTOR (utility) ───────────────────────────────
@app.post("/api/v1/extract/keywords")
async def extract_keywords(body: dict):
    """Fast keyword extraction from JD — no LLM needed"""
    jd = body.get("text", "")
    if not jd:
        raise HTTPException(status_code=400, detail="text required")

    # Common ATS keyword patterns
    tech_pattern = r'\b(Python|Java|SQL|AWS|Azure|GCP|React|Node|TypeScript|Docker|Kubernetes|ML|AI|API|REST|GraphQL|Git|Agile|Scrum|CI/CD|DevOps|MBA|PMP|CPA|CFA|Salesforce|SAP|Excel|PowerBI|Tableau)\b'
    soft_pattern = r'\b(leadership|communication|collaboration|management|strategy|analysis|problem.solving|cross.functional|stakeholder|mentoring)\b'

    tech_kw = list(set(re.findall(tech_pattern, jd, re.I)))
    soft_kw = list(set(re.findall(soft_pattern, jd, re.I)))

    return {
        "technical_keywords": tech_kw[:20],
        "soft_skills": soft_kw[:10],
        "total": len(tech_kw) + len(soft_kw)
    }
