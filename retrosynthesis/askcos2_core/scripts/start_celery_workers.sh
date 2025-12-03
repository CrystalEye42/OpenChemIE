IFS=","

for args in \
  2,context_recommender_worker \
  2,forward_worker \
  1,general_selectivity_worker \
  1,impurity_predictor_worker \
  1,pathway_ranker_worker \
  2,retro_worker \
  1,site_selectivity_worker \
  2,tree_analysis_worker \
  2,tree_search_expand_one_worker \
  2,tree_search_mcts_worker \
  2,tree_search_retro_star_worker \
  2,tree_search_worker; \
  do set -- $args;
  CONCURRENCY=$1;
  QUEUE_NAME=$2;
  WORKER_NAME=$2;
  celery -A askcos2_celery worker -D -c "$CONCURRENCY" -Q "$QUEUE_NAME" -n "$WORKER_NAME"@%h --pool=gevent;
  done;

celery -A askcos2_celery worker -c 20 -Q generic_worker -n generic_worker@%h --pool=gevent
