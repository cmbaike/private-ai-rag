#!/usr/bin/env python3
"""Apply SQL migrations in ``migrations/`` against DATABASE_URL.

Tracks applied files in ``schema_migrations`` so a fresh machine can run:

    uv run python scripts/migrate.py

Existing databases that were bootstrapped by hand can record the current
files as already applied without re-running them:

    uv run python scripts/migrate.py --stamp
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "migrations"
TRACKING_TABLE = "schema_migrations"


def migration_files() -> list[Path]:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No *.sql files in {MIGRATIONS_DIR}")
    return files


def ensure_tracking(conn: psycopg.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def applied_filenames(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute(f"SELECT filename FROM {TRACKING_TABLE}").fetchall()
    return {row[0] for row in rows}


def looks_previously_bootstrapped(conn: psycopg.Connection) -> bool:
    """True when app tables already exist (legacy init_db / hand-applied SQL)."""
    row = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = ANY (%s)
        )
        """,
        (
            [
                "documents",
                "chunks",
                "ask_audit",
                "investigations",
                "tool_execution_audit",
                "investigation_audit_events",
                "resource_health_checks",
            ],
        ),
    ).fetchone()
    return bool(row and row[0])


def apply_file(conn: psycopg.Connection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    with conn.transaction():
        conn.execute(sql)
        conn.execute(
            f"INSERT INTO {TRACKING_TABLE} (filename) VALUES (%s)",
            (path.name,),
        )


def stamp_file(conn: psycopg.Connection, path: Path) -> None:
    conn.execute(
        f"""
        INSERT INTO {TRACKING_TABLE} (filename)
        VALUES (%s)
        ON CONFLICT (filename) DO NOTHING
        """,
        (path.name,),
    )


def database_url() -> str:
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (configure .env or the environment).")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="Record all migration files as applied without executing them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pending migration filenames and exit",
    )
    args = parser.parse_args(argv)

    try:
        files = migration_files()
        url = database_url()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    with psycopg.connect(url) as conn:
        ensure_tracking(conn)
        conn.commit()
        done = applied_filenames(conn)
        pending = [path for path in files if path.name not in done]

        if args.dry_run:
            if not pending:
                print("No pending migrations.")
            else:
                print("Pending migrations:")
                for path in pending:
                    print(f"  {path.name}")
            return 0

        if args.stamp:
            with conn.transaction():
                for path in files:
                    stamp_file(conn, path)
            print(f"Stamped {len(files)} migration file(s).")
            return 0

        if not pending:
            print("Database schema is up to date.")
            return 0

        if not done and looks_previously_bootstrapped(conn):
            print(
                "Existing application tables were found, but schema_migrations is empty.\n"
                "Refusing to re-apply migrations that may not be idempotent.\n"
                "If this database already matches migrations/*.sql, run once:\n"
                "  uv run python scripts/migrate.py --stamp",
                file=sys.stderr,
            )
            return 2

        try:
            for path in pending:
                print(f"Applying {path.name} …")
                apply_file(conn, path)
                print(f"Applied {path.name}")
        except psycopg.Error as exc:
            print(f"Migration failed: {exc}", file=sys.stderr)
            print(
                "If this database was bootstrapped before schema_migrations existed, "
                "record the current files once with: uv run python scripts/migrate.py --stamp",
                file=sys.stderr,
            )
            return 1

    print(f"Applied {len(pending)} migration(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
