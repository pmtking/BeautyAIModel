"""
💬 Chat API v2 — چت‌بات دوستانه Buti
POST /api/v1/chat
  { user_id, text, has_photo?, last_result_ok?, use_llm? }
  → { status, reply, intent, is_edit_request, engine }
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from app.services.chat_bot import CHAT_BOT

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatIn(BaseModel):
    user_id: str = Field(..., min_length=1)
    text: str = Field(..., max_length=1000)
    has_photo: bool = False
    last_result_ok: Optional[bool] = None
    use_llm: bool = True


@router.post("/chat")
async def chat(body: ChatIn):
    try:
        r = CHAT_BOT.reply(
            body.user_id, body.text,
            has_photo=body.has_photo,
            last_result_ok=body.last_result_ok,
            use_llm=body.use_llm,
        )
        return {
            'status': 'success',
            'reply': r['reply'],
            'intent': r['intent'],
            'is_edit_request': r['is_edit_request'],
            'engine': r.get('engine', 'local'),
        }
    except Exception as e:
        logger.warning(f"chat failed: {e}")
        return {
            'status': 'success',
            'reply': None,           # اپ fallback محلی دارد
            'intent': 'chat',
            'is_edit_request': False,
            'engine': 'error',
        }


@router.get("/chat/health")
async def chat_health():
    return {'status': 'ok', 'service': 'buti-chat-v2'}
