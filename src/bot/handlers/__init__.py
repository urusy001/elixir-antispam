from .chat import router as chat_router
from .admin import router as admin_router
from .user import router as user_router

__all__ = ["chat_router", "admin_router", "user_router"]