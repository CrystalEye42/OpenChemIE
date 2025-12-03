import os
import re
import uuid
from configs import db_config
from fastapi import Depends, HTTPException, Response, status
from jose import JWTError, jwt
from keycloak import KeycloakOpenID
from keycloak import KeycloakAdmin
from keycloak.exceptions import KeycloakGetError
from pydantic import BaseModel
from pymongo import errors, MongoClient
from typing import Annotated, Any
from utils import register_util
from utils.custom_auth import CustomAuthAPI
from utils.oauth2 import (
    ALGORITHM,
    credentials_exception,
    oauth2_scheme,
    pwd_context,
    SECRET_KEY
)


class User(BaseModel):
    username: str
    hashed_password: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool = False
    is_superuser: bool = False
    last_login: str | None = None


@register_util(name="user_controller")
class UserController:
    prefixes = ["user"]
    methods_to_bind: dict[str, list[str]] = {
        "register": ["POST"],
        "update": ["POST"],
        "reset_password": ["POST"],
        "delete": ["DELETE"],
        "enable": ["GET"],
        "disable": ["GET"],
        "get_all_users": ["GET"],
        "get_current_user": ["GET"],
        "promote": ["GET"],
        "demote": ["GET"],
        "am_i_superuser": ["GET"],
        "update_last_login": ["POST"],
    }

    def __init__(self, util_config: dict[str, Any]):
        """
        Initialize database connection.
        """
        assert util_config["engine"] == "db", \
            "Only the 'db' engine is supported for user controller!"

        self.client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)

        database = util_config["database"]
        collection = util_config["collection"]

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb to load user data")
        else:
            self.collection = self.client[database][collection]
            self.db = self.client[database]

        # create default admin account if there's none
        users = self.collection.find()
        users = [User(**u) for u in users]
        if not any(u.is_superuser for u in users):
            self.register_superuser(
                username="askcos_admin",
                password="reallybadpassword"
            )

        server_url = os.environ.get("KEYCLOAK_SERVER_URL", "")
        realm_name = os.environ.get("KEYCLOAK_REALM_NAME", "")
        client_id = os.environ.get("KEYCLOAK_CLIENT_ID", "")
        client_secret_key = os.environ.get("KEYCLOAK_CLIENT_SECRET_KEY", "")

        if server_url:
            self.keycloak_openid = KeycloakOpenID(
                server_url=server_url,
                realm_name=realm_name,
                client_id=client_id,
                client_secret_key=client_secret_key,
                verify=os.environ.get("KEYCLOAK_VERIFY_SSL", "true").lower() != "false"
            )
            self.keycloak_public_key = "-----BEGIN PUBLIC KEY-----\n" + \
                                       self.keycloak_openid.public_key() + \
                                       "\n-----END PUBLIC KEY-----"

        custom_auth_url = os.environ.get("CUSTOM_AUTH_URL")
        if custom_auth_url:
            self.custom_auth_api = CustomAuthAPI(custom_auth_url)

    def get_user_by_name(self, username: str) -> User | None:
        query = {"username": username}
        try:
            user = self.collection.find_one(query)
            user = User(**user)
        except:
            user = None

        return user

    def get_current_user(self, token: Annotated[str, Depends(oauth2_scheme)]
                         ) -> User:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise JWTError
            user = self.get_user_by_name(username=username)

        except JWTError:
            # second try for keycloak
            try:
                options = {
                    "verify_signature": True,
                    "verify_aud": False,
                    "verify_exp": False
                }
                token_info = self.keycloak_openid.decode_token(
                    token=token,
                    key=self.keycloak_public_key,
                    options=options
                )
                # Derive a safe username from token claims (sanitize BEFORE lookup/register)
                raw_username = (
                    token_info.get("preferred_username")
                    or token_info.get("email")
                    or token_info.get("sub")
                    or "user"
                )
                # Sanitize to only allow [A-Za-z0-9_]
                safe_base = re.sub(r"[^A-Za-z0-9_]", "_", str(raw_username))
                # Collapse multiple underscores
                safe_base = re.sub(r"_+", "_", safe_base).strip("_")
                if not safe_base:
                    safe_base = "user"
                username = f"{safe_base}_sso"
                user = self.get_user_by_name(username=username)
                # automatically register in mongo if user doesn't exist
                if user is None:
                    self.register(
                        username=username,
                        password=str(uuid.uuid4())
                    )
                    user = self.get_user_by_name(username=username)
            except Exception as e:
                # Log the specific Keycloak error for diagnostics
                try:
                    print(f"Keycloak token validation failed: {repr(e)}")
                except Exception:
                    pass
                # Fallback: decode without signature verification (development only)
                try:
                    token_info = jwt.get_unverified_claims(token)
                    raw_username = (
                        token_info.get("preferred_username")
                        or token_info.get("email")
                        or token_info.get("sub")
                        or "user"
                    )
                    safe_base = re.sub(r"[^A-Za-z0-9_]", "_", str(raw_username))
                    safe_base = re.sub(r"_+", "_", safe_base).strip("_") or "user"
                    username = f"{safe_base}_sso"
                    user = self.get_user_by_name(username=username)
                    if user is None:
                        self.register(
                            username=username,
                            password=str(uuid.uuid4())
                        )
                        user = self.get_user_by_name(username=username)
                except Exception as e2:
                    try:
                        print(f"Unverified token decode failed: {repr(e2)}")
                    except Exception:
                        pass
                    # third try for custom auth
                # Optional custom auth provider support
                custom_auth = getattr(self, "custom_auth_api", None)
                if custom_auth is not None:
                    try:
                        username = custom_auth.decode(token)
                        if username is None:
                            raise credentials_exception
                        user = self.get_user_by_name(username=username)
                        # automatically register in mongo if user doesn't exist
                        if user is None:
                            self.register(
                                username=username,
                                password=str(uuid.uuid4())
                            )
                            user = self.get_user_by_name(username=username)
                    except Exception:
                        raise credentials_exception
                else:
                    # No custom auth configured and Keycloak validation failed
                    raise credentials_exception

        if user is None:
            raise credentials_exception

        if user.disabled:
            raise HTTPException(status_code=400, detail="Disabled user")

        return user

    def am_i_superuser(self, token: Annotated[str, Depends(oauth2_scheme)]
                       ) -> bool:
        user = self.get_current_user(token=token)

        return user.is_superuser

    def register(
        self,
        username: str,
        password: str,
        email: str = None,
        full_name: str = None,
        disabled: bool = False
    ) -> Response:
        if not username.replace('_', '').isalnum():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid username: {username}! "
                       f"Only numbers, letters and underscores are allowed.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if self.collection.find_one({"username": username}):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"username: {username} already exists!",
                headers={"WWW-Authenticate": "Bearer"},
            )

        hashed_password = pwd_context.hash(password)
        doc = {
            "username": username,
            "hashed_password": hashed_password,
            "email": email,
            "full_name": full_name,
            "disabled": disabled,
            "is_superuser": False
        }
        User(**doc)         # data validation, if any
        self.collection.insert_one(doc)

        return Response(content=f"Successfully register user: {username}!")

    def register_superuser(
        self,
        username: str,
        password: str,
    ) -> Response:
        hashed_password = pwd_context.hash(password)
        doc = {
            "username": username,
            "hashed_password": hashed_password,
            "email": None,
            "full_name": None,
            "disabled": False,
            "is_superuser": True
        }
        User(**doc)         
        self.collection.insert_one(doc)

        return Response(content=f"Successfully register superuser: {username}!")

    def update(
        self,
        username: str,
        email: str = None,
        full_name: str = None,
        disabled: bool = False,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        user = self.get_current_user(token)
        if user.username == username or user.is_superuser:
            self.collection.update_one(
                {"username": username},
                {"$set": {
                    "email": email,
                    "full_name": full_name,
                    "disabled": disabled
                }}
            )
            try:
                if hasattr(self, "keycloak_openid") and username.endswith("_sso"):
                    self._sync_keycloak_user(mongo_username=username, email=email, full_name=full_name,
                                             email_verified=True if email else None)
            except Exception as e:
                try:
                    print(f"Keycloak profile sync failed: {repr(e)}")
                except Exception:
                    pass

        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="update operation only permitted by the owner superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return Response(content=f"Successfully update records for {username}!")

    def update_last_login(
        self,
        username: str,
        last_login: str,
    ) -> Response:
        self.collection.update_one(
            {"username": username},
            {"$set": {
                "last_login": last_login
            }}
        )

        return Response(content=f"Successfully reset the password for {username}!")

    def reset_password(
        self,
        username: str,
        password: str,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        user = self.get_current_user(token)
        if user.username == username or user.is_superuser:
            hashed_password = pwd_context.hash(password)
            self.collection.update_one(
                {"username": username},
                {"$set": {
                    "hashed_password": hashed_password
                }}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="password reset only permitted by the owner or superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return Response(content=f"Successfully reset the password for {username}!")

    def promote(
        self,
        username: str,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        user = self.get_current_user(token)
        if user.is_superuser:
            self.collection.update_one(
                {"username": username},
                {"$set": {
                    "is_superuser": True
                }}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="make superuser only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return Response(content=f"Successfully reset the password for {username}!")

    def demote(
        self,
        username: str,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        user = self.get_current_user(token)
        if user.is_superuser:
            self.collection.update_one(
                {"username": username},
                {"$set": {
                    "is_superuser": False
                }}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="remove superuser only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return Response(content=f"Successfully reset the password for {username}!")

    def delete(
        self,
        username: str,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> Response:
        user = self.get_current_user(token)
        if not (user.username == username or user.is_superuser):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="delete operation only permitted by the owner or superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Enforce Keycloak→Mongo deletion
        if username.endswith("_sso"):
            # Require Keycloak admin to avoid orphaned KC accounts
            try:
                kc_admin = self._get_keycloak_admin()
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Keycloak admin not configured or unavailable; cannot delete SSO user. Deletion aborted."
                )

            # Resolve KC user id; treat not-found as already deleted
            try:
                kc_user_id = self._find_kc_user_id(kc_admin=kc_admin, mongo_username=username)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to resolve user in Keycloak; deletion aborted."
                )

            if kc_user_id:
                try:
                    kc_admin.delete_user(user_id=kc_user_id)
                except KeycloakGetError as e:
                    if getattr(e, "response_code", None) != 404:
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Failed to delete user in Keycloak; deletion aborted."
                        )
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Failed to delete user in Keycloak; deletion aborted."
                    )

        self.collection.delete_one({"username": username})

        return Response(content=f"Successfully delete user: {username}!")

    def enable(
        self,
        username: str,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> Response:
        user = self.get_current_user(token)
        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="enable operation only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )
        self.collection.update_one(
            {"username": username},
            {"$set": {"disabled": False}}
        )

        return Response(content=f"Successfully enable user: {username}!")

    def disable(
        self,
        username: str,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> Response:
        user = self.get_current_user(token)
        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="disable operation only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )
        self.collection.update_one(
            {"username": username},
            {"$set": {"disabled": True}}
        )

        return Response(content=f"Successfully disable user: {username}!")

    def get_all_users(self, token: Annotated[str, Depends(oauth2_scheme)]
                      ) -> list[User]:
        user = self.get_current_user(token)
        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="get_all_users operation only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )
        users = self.collection.find()
        users = [User(**u) for u in users]

        return users

    def _get_keycloak_admin(self):
        server_url = os.environ.get("KEYCLOAK_SERVER_URL", "").strip()
        if server_url and not server_url.endswith("/"):
            server_url = server_url + "/"
        realm_name = os.environ.get("KEYCLOAK_REALM_NAME", "").strip()
        verify = os.environ.get("KEYCLOAK_VERIFY_SSL", "true").lower() != "false"

        if not server_url or not realm_name:
            raise RuntimeError("KEYCLOAK_SERVER_URL/KEYCLOAK_REALM_NAME not configured for Keycloak admin")

        # Support both KEYCLOAK_ADMIN_USERNAME and KEYCLOAK_ADMIN (alias)
        admin_username = os.environ.get("KEYCLOAK_ADMIN_USERNAME") or os.environ.get("KEYCLOAK_ADMIN")
        admin_password = os.environ.get("KEYCLOAK_ADMIN_PASSWORD")
        admin_realm = os.environ.get("KEYCLOAK_ADMIN_REALM", "master")

        if admin_username and admin_password:
            return KeycloakAdmin(
                server_url=server_url,
                username=admin_username,
                password=admin_password,
                realm_name=realm_name,           # operate in the target realm (e.g., askcos)
                user_realm_name=admin_realm,     # admin authentication realm (often master)
                verify=verify,
            )

        admin_client_id = os.environ.get("KEYCLOAK_ADMIN_CLIENT_ID")
        admin_client_secret = os.environ.get("KEYCLOAK_ADMIN_CLIENT_SECRET")
        if admin_client_id and admin_client_secret:
            oid = KeycloakOpenID(
                server_url=server_url,
                realm_name=admin_realm,
                client_id=admin_client_id,
                client_secret_key=admin_client_secret,
                verify=verify,
            )
            token = oid.token(grant_type="client_credentials")
            return KeycloakAdmin(
                server_url=server_url,
                realm_name=realm_name,           # operate in target realm (e.g., askcos)
                token=token["access_token"],
                verify=verify,
            )

        raise RuntimeError("No Keycloak admin credentials configured")


    def _find_kc_user_id(self, kc_admin: KeycloakAdmin, mongo_username: str, email: str | None = None) -> str | None:
        kc_username = mongo_username[:-4] if mongo_username.endswith("_sso") else mongo_username
        # 1) Try by username
        try:
            users = kc_admin.get_users(query={"username": kc_username}) or []
            if users:
                return users[0].get("id")
        except Exception:
            pass
        # 2) Try by email from args or Mongo
        if email is None:
            doc = self.collection.find_one({"username": mongo_username}) or {}
            email = doc.get("email")
        if email:
            try:
                users = kc_admin.get_users(query={"email": email}) or []
                if users:
                    return users[0].get("id")
            except Exception:
                pass
        # 3) Broad search by search term (username/email), then filter client-side
        try:
            for term in filter(None, [kc_username, email]):
                users = kc_admin.get_users(query={"search": term}) or []
                for u in users:
                    if u.get("username") == kc_username or u.get("email") == email:
                        return u.get("id")
        except Exception:
            pass
        return None

    def _sync_keycloak_user(self, mongo_username: str, email: str | None, full_name: str | None,
                             email_verified: bool | None = None) -> None:
        kc_admin = self._get_keycloak_admin()
        kc_user_id = self._find_kc_user_id(kc_admin=kc_admin, mongo_username=mongo_username, email=email)
        if not kc_user_id:
            return
        payload: dict[str, Any] = {}
        if email:
            payload["email"] = email
        if full_name:
            try:
                parts = full_name.strip().split()
                if len(parts) >= 2:
                    payload["firstName"], payload["lastName"] = parts[0], " ".join(parts[1:])
                else:
                    payload["firstName"] = full_name
            except Exception:
                pass
        if email_verified is True:
            payload["emailVerified"] = True
        if payload:
            kc_admin.update_user(user_id=kc_user_id, payload=payload)