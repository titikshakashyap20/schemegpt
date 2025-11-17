from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# For MVP we default to SQLite. Replace with Postgres URL in production.
DATABASE_URL = "sqlite:///./schemes.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
