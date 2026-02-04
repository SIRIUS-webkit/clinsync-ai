"""HIPAA-style anonymization utilities."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Anonymizer:
    """Anonymize PHI fields in text and structured payloads."""

    salt: str

    def anonymize_text(self, text: str) -> str:
        """Redact common PHI patterns from free-form text."""
        if not text:
            return text

        redactions = {
            r"\b\d{3}-\d{2}-\d{4}\b": "[REDACTED_SSN]",
            r"\b\d{10}\b": "[REDACTED_PHONE]",
            r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b": "[REDACTED_PHONE]",
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}": "[REDACTED_EMAIL]",
            r"\bMRN[:\s]*\d+\b": "[REDACTED_MRN]",
            r"\bDOB[:\s]*\d{1,2}/\d{1,2}/\d{2,4}\b": "[REDACTED_DOB]",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b": "[REDACTED_DATE]",
        }
        anonymized = text
        for pattern, token in redactions.items():
            anonymized = re.sub(pattern, token, anonymized, flags=re.IGNORECASE)

        name_pattern = r"\b(Patient|Name|Pt)[:\s]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
        anonymized = re.sub(name_pattern, "[REDACTED_NAME]", anonymized)
        return anonymized

    def anonymize_fhir(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a FHIR payload with identifiers hashed."""
        def _hash(value: str) -> str:
            digest = hashlib.sha256(f"{self.salt}:{value}".encode("utf-8")).hexdigest()
            return f"hash:{digest}"

        redacted = payload.copy()
        if "identifier" in redacted and isinstance(redacted["identifier"], list):
            for identifier in redacted["identifier"]:
                if isinstance(identifier, dict) and "value" in identifier:
                    identifier["value"] = _hash(str(identifier["value"]))

        if "name" in redacted:
            redacted["name"] = "[REDACTED_NAME]"

        if "telecom" in redacted:
            redacted["telecom"] = "[REDACTED_TELECOM]"

        return redacted

    def hash_identifier(self, value: str) -> str:
        """Hash a single identifier for storage."""
        digest = hashlib.sha256(f"{self.salt}:{value}".encode("utf-8")).hexdigest()
        return f"hash:{digest}"
