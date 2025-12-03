from configs.module_config_full import module_config

# Default serializers
task_serializer = "json"
result_serializer = "json"

# Allowed content types - other message types are discarded
accept_content = ["json"]

# Timezone for message dates and times (set to match django settings)
timezone = "UTC"

# Convert dates and times to UTC
enable_utc = True

# Time that results remain in queue
result_expires = 1800  # 30 min
result_persistent = False

# Interval between sending worker heartbeats
broker_heartbeat = 0

# Maximum allowed priority (larger values are capped)
# Having more priority levels will consume more CPU resources
# Max priority of 2 enables 3 levels - 0, 1, 2
task_queue_max_priority = 2

# Default priority for tasks
task_default_priority = 1

# Track when tasks are started
task_track_started = True

# Number of messages to prefetch from queue
# Prefetching fewer tasks gives queue more opportunity to reorder by priority
worker_prefetch_multiplier = 1

# Modules to be imported by each worker
imports = ["askcos2_celery.tasks"]

# Task routes (to make sure workers are task-specific)
# Key is pattern matched against task name to determine queue
task_routes = {
    "askcos2_celery.tasks.base_task": {"queue": "generic_worker"},
    "askcos2_celery.tasks.rdkit*": {"queue": "generic_worker"},
    "askcos2_celery.tasks.legacy_task": {"queue": "generic_worker"}
}

for module, to_start in module_config["modules_to_start"].items():
    if to_start:
        if module.startswith("forward") or module.startswith("retro"):
            continue
        task_routes[f"askcos2_celery.tasks.{module}*"] = {"queue": f"{module}_worker"}

# Special treatment to route the tasks from controllers
task_routes[f"askcos2_celery.tasks.forward*"] = {"queue": "forward_worker"}
task_routes[f"askcos2_celery.tasks.retro*"] = {"queue": "retro_worker"}
task_routes[f"askcos2_celery.tasks.tree_analysis*"] = {"queue": "tree_analysis_worker"}
task_routes[f"askcos2_celery.tasks.tree_search_task"] = {"queue": "tree_search_worker"}

print(f"celery_imports: {imports}")
print("celery_task_routes:")
[print(f"{k}: {v}") for k, v in task_routes.items()]
