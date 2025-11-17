# placeholder for future SQLAlchemy models
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Scheme(Base):
    __tablename__ = "schemes"
    id = Column(Integer, primary_key=True)
    name = Column(String(256))
    authority = Column(String(256))
    source_url = Column(String(1024))
    raw_text = Column(Text)
