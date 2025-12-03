from datetime import datetime
from fastapi import Depends, Response
from pydantic import BaseModel, constr, Field
from typing import Annotated
from utils import register_util
from utils.base import BaseBanlistController
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry


class BlacklistedChemical(BaseModel):
    """
    Reimplemented BlacklistedChemical model
    """
    id: str = ""
    user: str
    description: constr(max_length=1000) = None
    created: datetime = Field(default_factory=datetime.now)
    dt: constr(max_length=200) = None
    smiles: constr(max_length=5000)
    active: bool = True


class BlacklistedChemicals(BaseModel):
    __root__: list[BlacklistedChemical]


@register_util(name="banned_chemicals")
class BannedChemicalsController(BaseBanlistController):
    """
    Util class for banned chemicals controller, alias for legacy BannedChemicalsViewSet
    """
    prefixes = ["banlist/chemicals"]
    methods_to_bind: dict[str, list[str]] = {
        "get": ["GET"],
        "post": ["POST"],
        "delete": ["DELETE"],
        "activate": ["GET"],
        "deactivate": ["GET"],
    }

    def get(self, token: Annotated[str, Depends(oauth2_scheme)]
            ) -> BlacklistedChemicals:
        """
        API endpoint for accessing banned chemicals.

        Method: GET

        Returns: list of banned chemicals belonging to the currently
            authenticated user
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {"user": user.username}
        cursor = self.collection.aggregate(
            [
                {"$match": query},
                {"$sort": {"dt": -1, "active": -1}}
            ]
        )
        banned_chemicals = []
        for r in cursor:
            banned_chemical = BlacklistedChemical(**r)
            banned_chemical.id = str(r.get("_id"))
            banned_chemicals.append(banned_chemical)

        banned_chemicals = BlacklistedChemicals(__root__=banned_chemicals)

        return banned_chemicals

    def post(
        self,
        description: str = "no description",
        dt: str = None,
        smiles: str = None,
        active: bool = True,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        """
        API endpoint for adding banned reactions.

        Method: POST
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)
        now = datetime.now()

        query = {
            "user": user.username,
            "smiles": smiles
        }

        if not self.collection.find_one(query):
            doc = {
                "user": user.username,
                "description": description,
                "created": now,
                "dt": dt or now.strftime("%B %d, %Y %H:%M:%S %p UTC"),
                "smiles": smiles,
                "active": active
            }
            BlacklistedChemical(**doc)              # data validation, if any
            self.collection.insert_one(doc)

            return Response(content=f"Successfully add banned chemicals for user: "
                                    f"{user.username}!")
        else:
            self.collection.update_one(
                query,
                {"$set": {
                    "description": description,
                    "dt": dt or now.strftime("%B %d, %Y %H:%M:%S %p UTC")
                }}
            )

            return Response(content=f"Successfully update banned chemicals for user: "
                                    f"{user.username}!")
