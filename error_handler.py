import streamlit as st
import duckdb
import logging
import requests
from functools import wraps

logger = logging.getLogger("ErrorHandler")

def safe_local_execution(func):
    """
    A decorator that wraps backend calls to catch Ollama connection failures 
    and DuckDB database errors, logging tracebacks and formatting UI messages.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        
        # 1. Handle Ollama Connection Errors
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama Connection Error in {func.__name__}: {e}", exc_info=True)
            st.error(
                "**Connection Refused:** Cannot connect to the local AI engine. "
                "Please ensure the Ollama application is running in the background."
            )
            return None
            
        # 2. Handle DuckDB Database Errors
        except duckdb.Error as e:
            logger.error(f"DuckDB Transaction Error in {func.__name__}: {e}", exc_info=True)
            st.error(
                "**Database Transaction Failed:** There was an issue writing to or reading from chat history.\n\n"
                f"Details: {str(e)}"
            )
            return None
            
        # 3. Handle Generic Unexpected Errors
        except Exception as e:
            logger.error(f"Unexpected System Error in {func.__name__}: {e}", exc_info=True)
            st.error(f"**Unexpected System Error:** {str(e)}")
            return None
            
    return wrapper