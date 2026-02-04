"""FHIR helper utilities."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def build_patient_resource(
    patient_id: str,
    name: str,
    birth_date: Optional[str] = None,
    gender: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a minimal FHIR Patient resource."""
    patient = {
        "resourceType": "Patient",
        "id": patient_id,
        "name": [{"text": name}],
    }
    if birth_date:
        patient["birthDate"] = birth_date
    if gender:
        patient["gender"] = gender
    return patient


def build_observation_resource(
    observation_id: str,
    patient_id: str,
    code: str,
    value: float,
    unit: str,
    issued: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a minimal FHIR Observation resource."""
    return {
        "resourceType": "Observation",
        "id": observation_id,
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {"text": code},
        "effectiveDateTime": issued or datetime.utcnow().isoformat(),
        "valueQuantity": {"value": value, "unit": unit},
    }
