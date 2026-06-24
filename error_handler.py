import streamlit as st
import duckdb
from functools import wraps
import requests

def safe_local_execution(func):
    """
    A decorator that wraps local backend calls to catch Ollama connection 
    failures and DuckDB database errors, formatting them for the Streamlit UI.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        
        # 1. Handle Ollama Connection Errors
        except requests.exceptions.ConnectionError:
            st.error(
                "**Connection Refused:** Cannot connect to the local AI engine. "
                "Please ensure the Ollama application is running in the background."
            )
            return None
            
        # 2. Handle DuckDB Database Errors
        except duckdb.Error as e:
            st.error(
                f"**Database Transaction Failed:** There was an issue writing to the chat history.\n\n"
                f"Details: {str(e)}"
            )
            return None
            
        # 3. Handle Generic Unexpected Errors
        except Exception as e:
            st.error(f"**Unexpected System Error:** {str(e)}")
            return None
            
    return wrapper