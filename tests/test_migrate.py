"""Migration runner and RAG schema bootstrap."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _load_migrate():
    spec = importlib.util.spec_from_file_location(
        "private_ai_migrate", ROOT / "scripts" / "migrate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_files_are_ordered_and_include_rag():
    module = _load_migrate()
    names = [path.name for path in module.migration_files()]
    assert names[0] == "000_rag_documents_and_chunks.sql"
    assert "001_create_ask_audit.sql" in names
    assert "009_resource_health_audit.sql" in names
    assert names == sorted(names)


def test_refuse_apply_on_legacy_database_without_tracking(monkeypatch):
    load_dotenv(ROOT / ".env")
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    module = _load_migrate()
    monkeypatch.setenv("DATABASE_URL", url)

    with psycopg.connect(url) as conn:
        module.ensure_tracking(conn)
        conn.execute(f"DELETE FROM {module.TRACKING_TABLE}")
        conn.commit()
        # Ensure at least one legacy table exists so the guard can fire.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ask_audit (
                id BIGSERIAL PRIMARY KEY,
                request_id UUID NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                answer TEXT,
                sufficient BOOLEAN,
                source_labels JSONB NOT NULL DEFAULT '[]'::jsonb,
                model TEXT,
                error_type TEXT
            )
            """
        )
        conn.commit()

    assert module.main([]) == 2
    assert module.main(["--stamp"]) == 0
    assert module.main([]) == 0


def test_apply_all_migrations_on_empty_database(monkeypatch):
    load_dotenv(ROOT / ".env")
    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL / TEST_DATABASE_URL is not configured")

    module = _load_migrate()
    parts = urlsplit(url)
    admin = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))
    db_name = f"private_ai_migrate_test_{os.getpid()}"
    scratch = urlunsplit((parts.scheme, parts.netloc, f"/{db_name}", "", ""))

    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{db_name}"')

    try:
        monkeypatch.setenv("DATABASE_URL", scratch)
        assert module.main([]) == 0
        assert module.main([]) == 0
        with psycopg.connect(scratch) as conn:
            applied = module.applied_filenames(conn)
            assert "000_rag_documents_and_chunks.sql" in applied
            assert "009_resource_health_audit.sql" in applied
            assert len(applied) == len(module.migration_files())
            assert conn.execute("SELECT to_regclass('public.documents')").fetchone()[0]
            assert conn.execute("SELECT to_regclass('public.investigations')").fetchone()[0]
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
