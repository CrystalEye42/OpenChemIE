import os
import requests


class CustomAuthAPI:
    def __init__(self, custom_auth_url: str):
        if custom_auth_url:
            self.custom_auth_url = custom_auth_url
        else:
            self.custom_auth_url = os.environ.get("CUSTOM_AUTH_URL")
        self.session = requests.Session()

    def decode(self, token: str | None = None) -> str | None:
        response = self.session.get(
            self.custom_auth_url,
            params={"token": token}
        )
        output = response.json()
        username = output.get("username")
        if username is not None:
            username = username + "_sso"

        return username
