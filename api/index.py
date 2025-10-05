"""Vercel serverless entrypoint for the FastAPI application."""

from __future__ import annotations

from api.serve import app as create_app

app = create_app()
