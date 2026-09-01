#!/usr/bin/env python3
"""
clean_failed_rows.py — Clean zero-token and failed/rate-limited rows from results database.

Usage:
    python clean_failed_rows.py --dry-run
    python clean_failed_rows.py
"""

import sys
import sqlite3
import argparse
from pathlib import Path
from src.config import RESULTS_DB
from src.database import export_to_json


def clean_database(dry_run: bool = True):
    if not Path(RESULTS_DB).exists():
        print(f"Database file not found at: {RESULTS_DB}")
        return

    conn = sqlite3.connect(RESULTS_DB, timeout=30.0)
    cursor = conn.cursor()

    # Query for failed rows: 0 tokens or explicit error strings
    select_query = """
        SELECT id, task_id, category, condition, score, tokens_total, error
        FROM results
        WHERE tokens_total = 0 
           OR tokens_total IS NULL
           OR (error IS NOT NULL AND error != '')
    """
    
    failed_rows = cursor.execute(select_query).fetchall()
    total_rows = cursor.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    print(f"\n--- Multimodel Database Cleaning Utility ---")
    print(f"Database Path: {RESULTS_DB}")
    print(f"Total Rows in DB: {total_rows}")
    print(f"Failed / Zero-Token Rows Found: {len(failed_rows)}")

    if not failed_rows:
        print("No failed or zero-token rows found. Database is clean!")
        conn.close()
        return

    print("\nSample Failed Rows to be Cleaned:")
    for r in failed_rows[:10]:
        row_id, task_id, cat, cond, score, tokens, err = r
        err_snippet = (err[:30] + "...") if err else "None"
        print(f"  [ID {row_id}] Task: {task_id} | Category: {cat} | Condition: {cond} | Tokens: {tokens} | Score: {score} | Error: {err_snippet}")
    
    if len(failed_rows) > 10:
        print(f"  ... and {len(failed_rows) - 10} more rows.")

    if dry_run:
        print("\n[DRY RUN MODE] No changes were made to the database.")
        print("Run 'python clean_failed_rows.py' (without --dry-run) to permanently delete these failed rows.")
    else:
        delete_query = """
            DELETE FROM results
            WHERE tokens_total = 0 
               OR tokens_total IS NULL
               OR (error IS NOT NULL AND error != '')
        """
        cursor.execute(delete_query)
        conn.commit()
        
        remaining_rows = cursor.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        print(f"\n[SUCCESS] Deleted {len(failed_rows)} failed rows.")
        print(f"Remaining Valid Rows in DB: {remaining_rows}")
        
        # Export updated DB state to JSON
        conn.close()
        try:
            export_to_json()
        except Exception as e:
            print(f"[Notice] Export to JSON warning: {e}")
        return

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Clean failed and 0-token rows from results database.")
    parser.add_argument("--dry-run", action="store_true", help="Preview rows to be deleted without modifying the DB.")
    args = parser.parse_args()

    clean_database(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
