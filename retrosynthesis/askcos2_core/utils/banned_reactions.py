from datetime import datetime
from fastapi import Depends, Response
from pydantic import BaseModel, constr, Field
from typing import Annotated
from utils import register_util
from utils.base import BaseBanlistController
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry


class BlacklistedReaction(BaseModel):
    """
    Reimplemented BlacklistedReaction model
    """
    id: str = ""
    user: str
    description: constr(max_length=1000) = None
    created: datetime = Field(default_factory=datetime.now)
    dt: constr(max_length=200) = None
    smiles: constr(max_length=5000)
    active: bool = True


class BlacklistedReactions(BaseModel):
    __root__: list[BlacklistedReaction]


@register_util(name="banned_reactions")
class BannedReactionsController(BaseBanlistController):
    """
    Util class for banned reactions controller, alias for legacy BannedReactionsViewSet
    """
    prefixes = ["banlist/reactions"]
    methods_to_bind: dict[str, list[str]] = {
        "get": ["GET"],
        "post": ["POST"],
        "delete": ["DELETE"],
        "activate": ["GET"],
        "deactivate": ["GET"],
    }

    def get(self, token: Annotated[str, Depends(oauth2_scheme)]
            ) -> BlacklistedReactions:
        """
        API endpoint for accessing banned reactions.

        Method: GET

        Returns: list of banned reactions belonging to the currently
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
        banned_reactions = []
        for r in cursor:
            banned_reaction = BlacklistedReaction(**r)
            banned_reaction.id = str(r.get("_id"))
            banned_reactions.append(banned_reaction)

        banned_reactions = BlacklistedReactions(__root__=banned_reactions)

        return banned_reactions

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
            BlacklistedReaction(**doc)              # data validation, if any
            self.collection.insert_one(doc)

            return Response(content=f"Successfully add banned reactions for user: "
                                    f"{user.username}!")
        else:
            self.collection.update_one(
                query,
                {"$set": {
                    "description": description,
                    "dt": dt or now.strftime("%B %d, %Y %H:%M:%S %p UTC")
                }}
            )

            return Response(content=f"Successfully update banned reactions for user: "
                                    f"{user.username}!")
