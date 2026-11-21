"""Backend admin commands.

    python -m backend.manage create-user admin
    python -m backend.manage list-users
In Docker:  docker compose exec backend python -m backend.manage create-user admin
"""
import getpass
import sys

from sqlalchemy import select

from backend.app import create_user, rows
from backend.db import users


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "create-user":
        password = getpass.getpass(f"Password for {sys.argv[2]}: ")
        if len(password) < 6:
            sys.exit("Use at least 6 characters.")
        if getpass.getpass("Repeat password: ") != password:
            sys.exit("Passwords don't match.")
        try:
            create_user(sys.argv[2], password)
        except ValueError as e:
            sys.exit(str(e))
        print(f"User {sys.argv[2]} created.")
    elif len(sys.argv) == 2 and sys.argv[1] == "list-users":
        for u in rows(select(users.c.username, users.c.created)):
            print(f"{u['username']}  (created {u['created']})")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
