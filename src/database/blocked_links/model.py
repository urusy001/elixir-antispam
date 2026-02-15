from sqlalchemy import Column, String

from src.database import Base


class BlockedLink(Base):
    __tablename__ = "blocked_links"

    url = Column(String, primary_key=True, nullable=False)
