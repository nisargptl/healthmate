#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import psycopg2
from psycopg2 import sql
import os
import sys

def create_database_if_not_exists():
    try:
        db_name = os.getenv('DATABASE_NAME', 'healthmate_db')
        db_user = os.getenv('DATABASE_USER', 'postgres')
        db_password = os.getenv('DATABASE_PASSWORD', 'root')
        db_host = os.getenv('DATABASE_HOST', 'localhost')
        db_port = os.getenv('DATABASE_PORT', '5432')

        conn = psycopg2.connect(
            dbname='postgres',
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute(sql.SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s"), [db_name])
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(db_name)))
            print(f'Successfully created database {db_name}')
        else:
            print(f'Database {db_name} already exists')

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthmate.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
