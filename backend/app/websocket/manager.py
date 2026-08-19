# backend/app/websocket/manager.py
"""
مدیریت اتصالات WebSocket
"""

from typing import Dict, List, Optional
from fastapi import WebSocket
import json
import logging
import time

logger = logging.getLogger(__name__)


class ConnectionManager:
    """مدیریت اتصالات WebSocket"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}
        self.connection_times: Dict[str, float] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """پذیرش اتصال جدید"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_sessions[user_id] = user_id
        self.connection_times[user_id] = time.time()
        logger.info(f"✅ User {user_id} connected. Total: {len(self.active_connections)}")

    def disconnect(self, user_id: str):
        """قطع اتصال"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        if user_id in self.connection_times:
            duration = time.time() - self.connection_times[user_id]
            logger.info(f"❌ User {user_id} disconnected (duration: {duration:.1f}s)")
            del self.connection_times[user_id]
        logger.info(f"Total connections: {len(self.active_connections)}")

    async def send_message(self, user_id: str, message: dict) -> bool:
        """ارسال پیام به یک کاربر خاص"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
                return True
            except Exception as e:
                logger.error(f"Error sending to {user_id}: {e}")
                self.disconnect(user_id)
        return False

    async def broadcast(self, message: dict, exclude: Optional[List[str]] = None):
        """ارسال به همه کاربران"""
        exclude = exclude or []
        for user_id, connection in self.active_connections.items():
            if user_id in exclude:
                continue
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {user_id}: {e}")
                self.disconnect(user_id)

    def get_active_users(self) -> List[str]:
        """لیست کاربران آنلاین"""
        return list(self.active_connections.keys())

    def is_connected(self, user_id: str) -> bool:
        """بررسی وضعیت اتصال کاربر"""
        return user_id in self.active_connections

    def get_connection_count(self) -> int:
        """تعداد اتصالات فعال"""
        return len(self.active_connections)

    def get_user_sessions(self) -> Dict:
        """دریافت اطلاعات جلسات"""
        return {
            'total': len(self.active_connections),
            'users': list(self.active_connections.keys()),
            'connection_times': self.connection_times
        }


# نمونه سراسری
manager = ConnectionManager()