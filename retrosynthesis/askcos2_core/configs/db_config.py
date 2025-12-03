import os

DATABASE = "askcos"

MONGO = {
    "host": os.environ.get("MONGO_HOST", "0.0.0.0"),
    "port": int(os.environ.get("MONGO_PORT", 27017)),
    "username": os.environ.get("MONGO_USER"),
    "password": os.environ.get("MONGO_PW"),
    "authSource": os.environ.get("MONGO_AUTH_DB", "admin"),
    "connect": False,
}
