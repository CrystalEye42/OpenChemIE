from pydantic import Field
from schemas.base import LowerCamelAliasModel
from typing import Literal


class ClusterSetting(LowerCamelAliasModel):
    feature: Literal["original", "outcomes", "all"] = "original"
    cluster_method: Literal["rxn_class", "hdbscan", "kmeans"] = "hdbscan"
    fp_type: Literal["morgan"] = Field(
        default="morgan",
        description="fingerprint type"
    )
    fp_length: int = Field(
        default=512,
        description="fingerprint bits"
    )
    fp_radius: int = Field(
        default=1,
        description="fingerprint radius"
    )
    classification_threshold: float = Field(
        default=0.2,
        description="threshold to classify a reaction as unknown when "
                    "clustering by 'rxn_class'"
    )
