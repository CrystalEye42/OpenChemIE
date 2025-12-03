import importlib
import os
from pydantic import Field, validator
from schemas.base import LowerCamelAliasModel
from typing import Any, Literal


class RetroBackendOption(LowerCamelAliasModel):
    retro_backend: Literal[
        "augmented_transformer",
        "exact_match",
        "graph2smiles",
        "template_enumeration",
        "template_relevance",
        "retrosim"
    ] = Field(
        default="template_relevance",
        description="backend for one-step retrosynthesis"
    )
    retro_model_name: str = Field(
        default="reaxys",
        description="backend model name for one-step retrosynthesis"
    )

    max_num_templates: int = Field(
        default=1000,
        description="number of templates to consider"
    )
    max_cum_prob: float = Field(
        default=0.995,
        description="maximum cumulative probability of templates"
    )
    attribute_filter: list[dict[str, Any]] = Field(
        default_factory=list,
        description="template attribute filter to apply before template application",
        example=[]
    )
    threshold: float = Field(
        default=0.3,
        description="threshold for similarity; "
                    "used for retrosim only"
    )
    top_k: int = Field(
        default=10,
        description="filter for k results returned; "
                    "used for retrosim only"
    )

    @validator("retro_model_name")
    def check_retro_model_name(cls, v, values):
        if "retro_backend" not in values:
            raise ValueError("retro_backend not supplied!")

        default_path = "configs.module_config_full"
        config_path = os.environ.get(
            "MODULE_CONFIG_PATH", default_path
        ).replace("/", ".").rstrip(".py")
        module_config = importlib.import_module(config_path).module_config

        retro_backend = values["retro_backend"]
        available_model_names = module_config[
            f"retro_{retro_backend}"]["deployment"]["available_model_names"]
        if v not in available_model_names:
            raise ValueError(f"Unsupported retro_model_name {v} for {retro_backend}")

        return v
