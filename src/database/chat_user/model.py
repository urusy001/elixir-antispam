from sqlalchemy import BigInteger, Column, Boolean, String, DateTime
from src.database import Base


class ChatUser(Base):
    __tablename__ = "chat_users"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=False, nullable=False)
    full_name = Column(String, nullable=False)
    username = Column(String, nullable=True, default=None)

    passed_poll = Column(Boolean, nullable=False, default=True)
    whitelist = Column(Boolean, nullable=False, default=False)

    muted_until = Column(DateTime(timezone=True), nullable=True, default=None)
    times_muted = Column(BigInteger, nullable=False, default=0)

    banned_until = Column(DateTime(timezone=True), nullable=True, default=None)
    times_banned = Column(BigInteger, nullable=False, default=0)

    messages_sent = Column(BigInteger, nullable=False, default=0)
    times_reported = Column(BigInteger, nullable=False, default=0)
    accused_spam = Column(Boolean, nullable=False, default=False)
    last_accused_text = Column(String, nullable=True, default=None)

    # 👇 состояние капчи
    poll_attempts = Column(BigInteger, nullable=False, default=0)
    poll_active = Column(Boolean, nullable=False, default=False)
    poll_message_id = Column(BigInteger, nullable=True, default=None)
    poll_chat_id = Column(BigInteger, nullable=True, default=None)
    poll_id = Column(String, nullable=True, default=None)
    poll_correct_option_id = Column(BigInteger, nullable=True, default=None)
