import os
from celery import Celery

# Define readable names for celery workers for status reporting
READABLE_NAMES = {
    "generic_worker":
        "Generic Worker (ML Router equivalent)",
    "context_recommender_worker":
        "Context Recommender Worker for V1 and V2",
    "count_analogs_worker":
        "Reaction Graph Enumeration Worker",
    "forward_worker":
        "Forward Predictor Worker",
    "general_selectivity_worker":
        "General Selectivity Worker",
    "impurity_predictor_worker":
        "Impurity Prediction Worker",
    "pathway_ranker_worker":
        "Pathway Ranker Worker",
    "retro_worker":
        "One-Step Retrosynthesis Worker",
    "site_selectivity_worker":
        "Site Selectivity Worker",
    "tree_analysis_worker":
        "Tree Analysis Worker",
    "tree_search_expand_one_worker":
        "IPP One-Step Expansion Worker",
    "tree_search_mcts_worker":
        "MCTS Worker (Tree Builder Coordinator equivalent)",
    "tree_search_retro_star_worker":
        "Retro* Worker (Tree Builder Coordinator equivalent)",
    "tree_search_worker":
        "Generic Worker for Tree Search"
}

# Note: cannot use guest for authenticating with broker unless on localhost
redis_host = os.environ.get("REDIS_HOST", "0.0.0.0")
redis_port = os.environ.get("REDIS_PORT", "6379")
redis_password = os.environ.get("REDIS_PASSWORD", "")
if redis_password:
    redis_password = ":{0}@".format(redis_password)
redis_url = "redis://{password}{host}:{port}".format(
    password=redis_password,
    host=redis_host,
    port=redis_port,
)

rabbit_host = os.environ.get("RABBITMQ_HOST", "0.0.0.0")
rabbit_port = os.environ.get("RABBITMQ_PORT", "5672")
rabbit_username = os.environ.get("RABBITMQ_USERNAME", "guest")
rabbit_password = os.environ.get("RABBITMQ_PASSWORD", "guest")
rabbit_url = "amqp://{username}:{password}@{host}:{port}/".format(
    username=rabbit_username,
    password=rabbit_password,
    host=rabbit_host,
    port=rabbit_port,
)

app = Celery(
    main="askcos2_celery",
    broker=rabbit_url,
    backend=redis_url,
)

# Using a string here means the worker don't have to serialize
# the configuration object to child processes.
app.config_from_object("askcos2_celery.celery_config")


if __name__ == "__main__":
    app.start()
