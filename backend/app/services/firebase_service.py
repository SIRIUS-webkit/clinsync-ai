import logging
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore, auth
from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def initialize_firebase() -> None:
    """Initialize Firebase Admin SDK."""
    try:
        # check if already initialized
        if firebase_admin._apps:
            return

        cred_path = Path(settings.firebase_credentials_path)

        # If path is relative, make it absolute ensuring it's relative to backend root or just use as is
        # Assuming run from backend/ directory
        if not cred_path.is_absolute():
            # Try to find it in current working directory which should be backend root
            cred_path = Path.cwd() / settings.firebase_credentials_path

        if not cred_path.exists():
            logger.warning(
                f"Firebase credentials not found at {cred_path}. Firebase integration disabled."
            )
            return

        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")

    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        # In a strict production environment, you might want to raise here
        # raise e


def get_firestore_client():
    """Get Firestore client."""
    try:
        return firestore.client()
    except ValueError:
        logger.warning("Firebase not active, returning None for Firestore client")
        return None


def verify_token(token: str):
    """Verify Firebase ID token."""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return None


def save_chat_message(
    patient_id: str,
    message_data: dict,
    response_data: dict,
) -> str | None:
    """Save chat record to Firestore."""
    db = get_firestore_client()
    if not db:
        return None

    try:
        # Use a subcollection: users/{patient_id}/chat_history
        # If patient_id is not provided, we might want to store in an 'anonymous' collection or similar,
        # but the caller works out the ID.

        # Create a document in the chat history
        doc_ref = (
            db.collection("users")
            .document(patient_id)
            .collection("chat_history")
            .document()
        )

        record = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "message": message_data,
            "response": response_data,
        }

        doc_ref.set(record)
        logger.debug(f"Saved chat record {doc_ref.id} for patient {patient_id}")
        return doc_ref.id

    except Exception as e:
        logger.error(f"Failed to save chat record: {e}")
        return None
