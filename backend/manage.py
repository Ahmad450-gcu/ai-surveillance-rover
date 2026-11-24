"""Backend admin commands.

    python -m backend.manage create-user admin
    python -m backend.manage set-password admin
    python -m backend.manage list-users
In Docker:  docker compose exec backend python -m backend.manage create-user admin
"""
import getpass
import sys

from sqlalchemy import select, update

from backend.app import create_user, execute, hash_password, rows
from backend.db import users


def ask_password(username):
    password = getpass.getpass(f"Password for {username}: ")
    if len(password) < 6:
        sys.exit("Use at least 6 characters.")
    if getpass.getpass("Repeat password: ") != password:
        sys.exit("Passwords don't match.")
    return password


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "create-user":
        password = ask_password(sys.argv[2])
        try:
            create_user(sys.argv[2], password)
        except ValueError as e:
            sys.exit(str(e))
        print(f"User {sys.argv[2]} created.")
    elif len(sys.argv) >= 3 and sys.argv[1] == "set-password":
        password = ask_password(sys.argv[2])
        changed = execute(update(users).where(users.c.username == sys.argv[2])
                          .values(pw_hash=hash_password(password))).rowcount
        if not changed:
            sys.exit(f"No user {sys.argv[2]}.")
        print(f"Password of {sys.argv[2]} changed.")
    elif len(sys.argv) == 2 and sys.argv[1] == "list-users":
        for u in rows(select(users.c.username, users.c.created)):
            print(f"{u['username']}  (created {u['created']})")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
