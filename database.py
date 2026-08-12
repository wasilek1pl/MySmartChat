import uuid
import duckdb
from datetime import datetime
import config
import logging

# Initialize the logger for database transactions
logger = logging.getLogger("Database")

def init_database():
    """Initializes DuckDB tables with explicit cursors for state stability."""
    try:
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id VARCHAR PRIMARY KEY,
                    title VARCHAR,
                    created_at TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id VARCHAR PRIMARY KEY,
                    chat_id VARCHAR,
                    role VARCHAR,
                    content VARCHAR,
                    timestamp TIMESTAMP
                )
            """)
            conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DuckDB tables: {e}", exc_info=True)

def save_message_to_db(chat_id, role, content):
    """Logs individual message entries and catches SQL lock errors."""
    msg_id = str(uuid.uuid4())  # Generate ID beforehand so the logger can safely reference it
    try:
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM chats WHERE chat_id = ?", [chat_id])
            if not cursor.fetchone():
                cursor.execute("INSERT INTO chats VALUES (?, ?, ?)", [chat_id, "General Discussion", datetime.now()])
            
            cursor.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", [msg_id, chat_id, role, content, datetime.now()])
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to save message {msg_id} to chat {chat_id}: {e}", exc_info=True)

def load_messages_from_db(chat_id):
    """Retrieves chronological array of message dictionaries matching a session token."""
    try:
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", [chat_id])
            rows = cursor.fetchall()
            return [{"role": row[0], "content": row[1]} for row in rows] if rows else []
    except Exception as e:
        logger.error(f"Failed to load messages for chat session {chat_id}: {e}", exc_info=True)
        return []

def update_chat_title(chat_id, new_title):
    """Commits title string updates to the target session row."""
    try:
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE chats SET title = ? WHERE chat_id = ?", [new_title, chat_id])
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to update title for chat {chat_id}: {e}", exc_info=True)

def get_chat_title(chat_id):
    """Fetches descriptive title for active container presentation safely."""
    if not chat_id:
        return "New Chat Session"
    try:
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM chats WHERE chat_id = ?", [chat_id])
            row = cursor.fetchone()
            return row[0] if row else "General Discussion"
    except Exception as e:
        logger.error(f"Failed to fetch title for chat {chat_id}: {e}", exc_info=True)
        return "General Discussion"

def delete_chat_from_db(chat_id):
    """Purges historical dependencies and parent records from database files."""
    try:
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE chat_id = ?", [chat_id])
            cursor.execute("DELETE FROM chats WHERE chat_id = ?", [chat_id])
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to purge chat {chat_id} from database: {e}", exc_info=True)

def get_all_chats():
    """Helper to fetch all historical chats for sidebar routing."""
    try:
        with duckdb.connect(config.DUCKDB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chat_id, title FROM chats ORDER BY created_at DESC")
            return cursor.fetchall() or []
    except Exception as e:
        logger.error(f"Failed to fetch historical chat lists: {e}", exc_info=True)
        return []