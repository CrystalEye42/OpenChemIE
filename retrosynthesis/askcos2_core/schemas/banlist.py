from datetime import datetime
from pydantic import BaseModel, constr, Field


class BlacklistedEntry(BaseModel):
    id: str = ""
    user: str
    description: constr(max_length=1000) = None
    created: datetime = Field(default_factory=datetime.now)
    dt: constr(max_length=200) = None
    smiles: constr(max_length=5000)
    active: bool = True


class BlacklistedChemicals(BaseModel):
    __root__: list[BlacklistedEntry]


class BlacklistedReactions(BaseModel):
    __root__: list[BlacklistedEntry]
