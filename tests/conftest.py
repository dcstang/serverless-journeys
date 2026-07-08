"""Shared pytest fixtures for the serverless-journeys test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_patient() -> dict:
    """A minimal patient dict sufficient for prompt-building functions."""
    return {
        "person_id": "test-person-1",
        "full_name": "Jordan Smith",
        "age": 65,
        "sex": "Male",
        "past_medical_history": ["Hypertension"],
    }
