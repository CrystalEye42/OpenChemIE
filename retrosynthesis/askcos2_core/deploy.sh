#!/usr/bin/env bash

################################################################################
#
#   ASKCOS Deployment Utilities
#
#    ~ To streamline deployment commands ~
#
################################################################################

set -e  # exit with nonzero exit code if anything fails

# Create environment variable files from examples if they don't exist
if [ ! -f ".env" ]; then
  cp .env.example .env
fi

# Get docker compose variables from .env
set -a
source .env
set +a

# Default argument values
BUYABLES=""
CHEMICALS=""
REACTIONS=""
RETRO_TEMPLATES=""
FORWARD_TEMPLATES=""
SELEC_REFERENCES=""
DB_DROP=""
DROP_INDEXES=false
LOCAL=false
BACKUP_DIR=""
IGNORE_DIFF=false

COMMANDS=""
while (( "$#" )); do
  case "$1" in
    -h|--help|help)
      usage
      exit
      ;;
    -f|--compose-file)
      if [ -z "$COMPOSE_FILE" ]; then
        COMPOSE_FILE=$2
      else
        COMPOSE_FILE=$COMPOSE_FILE:$2
      fi
      shift 2
      ;;
    -p|--project-name)
      COMPOSE_PROJECT_NAME=$2
      shift 2
      ;;
    -l|--local)
      LOCAL=true
      shift 1
      ;;
    -v|--version)
      VERSION_NUMBER=$2
      shift 2
      ;;
    -b|--buyables)
      BUYABLES=$2
      shift 2
      ;;
    -c|--chemicals)
      CHEMICALS=$2
      shift 2
      ;;
    -x|--reactions)
      REACTIONS=$2
      shift 2
      ;;
    -r|--retro-templates)
      RETRO_TEMPLATES=$2
      shift 2
      ;;
    -t|--forward-templates)
      FORWARD_TEMPLATES=$2
      shift 2
      ;;
    -e|--selec-references)
      SELEC_REFERENCES=$2
      shift 2
      ;;
    -a|--append)
      echo 'The -a,--append argument is no longer necessary. Documents are appended by default.'
      shift 1
      ;;
    -z|--drop)
      DB_DROP="--drop"
      shift 1
      ;;
    -i|--drop-indexes)
      DROP_INDEXES=true
      shift 1
      ;;
    -n|--ignore-diff)
      IGNORE_DIFF=true
      shift 1
      ;;
    -d|--backup-directory)
      BACKUP_DIR=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*) # any other flag
      echo "Error: Unsupported flag $1" >&2  # print to stderr
      exit 1
      ;;
    *) # preserve positional arguments
      COMMANDS="$COMMANDS $1"
      shift
      ;;
  esac
done

# Set positional arguments in their proper place
eval set -- "$COMMANDS"

# Export variables needed by docker compose
export VERSION_NUMBER
export COMPOSE_FILE
export COMPOSE_PROJECT_NAME

# Define various functions
diff-env() {
  # V2 update: dropped customization from diff since Django is no longer used
  if [ "$IGNORE_DIFF" = "true" ]; then
    return 0
  fi

  output1="$(diff -u .env .env.example)" || true
  if [ -n "$output1" ]; then
    echo -e "\033[91m*** WARNING ***\033[00m"
    echo "Local .env file is different from .env.example"
    echo "This could be due to changes to the example or local changes."
    echo "Please review the diff to determine if any changes are necessary:"
    echo
    echo "$output1"
    echo
  fi

  if [ -n "$output1" ]; then
    echo "Local configuration files differ from examples (see above)! What would you like to do?"
    echo "  (c)ontinue without changes  (use -n flag to skip prompt and continue in the future)"
    echo "  (o)verwrite your local files with the examples  (the above diff(s) will be applied)"
    echo "  (s)top and make changes manually"
    read -rp '>>> ' response
    case "$response" in
      [Cc])
        echo "Continuing without changes."
        ;;
      [Oo])
        echo "Overwriting local files."
        cp .env.example .env
        source .env
        ;;
      [Ss])
        echo "Stopping."
        exit 1
        ;;
      *)
        echo "Unrecognized option. Stopping."
        exit 1
        ;;
    esac
  fi
}

copy-nginx-conf() {
  # always create the cert if not exist, even for http
  # the cert and key files are volume mapped by docker compose, so creating them as files is more idiot-proof
  # otherwise they might be created as directories
  if [ ! -f "askcos.ssl.cert" ]; then
    echo "Creating self-signed SSL certificates."
    openssl req -new -newkey rsa:4096 -days 3650 -nodes -x509 -subj "/C=US/ST=MA/L=BOS/O=askcos/CN=askcos.$RANDOM.com" -keyout askcos.ssl.key -out askcos.ssl.cert
    echo
  fi

  if [ -f "nginx.conf" ]; then
    echo "Found nginx config at ./nginx.conf"
    return
  fi

  if [ "$PROTOCOL" = "http" ]; then
    echo "Using default http nginx configuration."
    cp nginx.http.conf nginx.conf
    echo
  elif [ "$PROTOCOL" = "https" ]; then
    echo "Using default https nginx configuration."
    cp nginx.https.conf nginx.conf
    echo
  fi
}

set-db-defaults() {
  # Set default values for seeding database if values are not already defined
  BUYABLES=${BUYABLES:-default}
  RETRO_TEMPLATES=${RETRO_TEMPLATES:-default}
  FORWARD_TEMPLATES=${FORWARD_TEMPLATES:-default}
  CHEMICALS=${CHEMICALS:-default}
  REACTIONS=${REACTIONS:-default}
  SELEC_REFERENCES=${SELEC_REFERENCES:-default}
}

run-mongo-js() {
  # arg 1 is js command
  docker compose -f compose.yaml exec -T mongo \
    bash -c 'mongosh --username ${MONGO_USER} --password ${MONGO_PW} --authenticationDatabase admin ${MONGO_HOST}/askcos --quiet --eval '"'$1'"
}

mongoimport() {
  # arg 1 is collection name
  # arg 2 is file path
  # arg 3 is a flag to pass to docker compose exec, e.g. -d to detach
  docker compose -f compose.yaml exec -T $3 mongo \
    bash -c 'gunzip -c '$2' | mongoimport --host ${MONGO_HOST} --username ${MONGO_USER} --password ${MONGO_PW} --authenticationDatabase admin --db askcos --collection '$1' --type json --jsonArray --numInsertionWorkers 8 '${DB_DROP}
}

mongoimport_upsert() {
  # arg 1 is collection name
  # arg 2 is file path
  # arg 3 is a flag to pass to docker compose exec, e.g. -d to detach
  docker compose -f compose.yaml exec -T $3 mongo \
    bash -c 'gunzip -c '$2' | mongoimport --host ${MONGO_HOST} --username ${MONGO_USER} --password ${MONGO_PW} --authenticationDatabase admin --db askcos --collection '$1' --type json --jsonArray --numInsertionWorkers 8 --mode upsert '${DB_DROP}
}

mongoexport() {
  # arg 1 is collection name
  # arg 2 is file path
  # arg 3 is a flag to pass to docker compose exec, e.g. -d to detach
  docker compose -f compose.yaml exec -T $3 mongo \
    bash -c 'mongoexport --host ${MONGO_HOST} --username ${MONGO_USER} --password ${MONGO_PW} --authenticationDatabase admin --db askcos --collection '$1' --type json --jsonArray | gzip > '$2
}

precompute() {
  # arg 1 is collection name, i.e., mode for precomputation
  docker compose -f compose.yaml exec -T precompute \
    /opt/conda/bin/python -m scripts.pre_compute --mode="$1"
}

index-db() {
  if [ "$DROP_INDEXES" = "true" ]; then
    echo "Dropping existing indexes in mongo database..."
    run-mongo-js 'db.buyables.dropIndexes()'
    run-mongo-js 'db.chemicals.dropIndexes()'
    run-mongo-js 'db.reactions.dropIndexes()'
    run-mongo-js 'db.retro_templates.dropIndexes()'
    run-mongo-js 'db.sites_refs.dropIndexes()'
  fi
  echo "Adding indexes to mongo database..."
  run-mongo-js 'db.buyables.createIndex({smiles: 1, source: 1})'
  run-mongo-js 'db.chemicals.createIndex({smiles: 1, template_set: 1})'
  run-mongo-js 'db.reactions.createIndex({reaction_id: 1, template_set: 1})'
  run-mongo-js 'db.retro_templates.createIndex({index: 1, template_set: 1})'
  run-mongo-js 'db.sites_refs.createIndexes([{index: 1}, {reactant: 1}])'
  echo "Indexing complete."
  echo
}

count-mongo-docs() {
  echo "Buyables collection:          $(run-mongo-js "db.buyables.estimatedDocumentCount({})" | tr -d '\r') / 509931 expected (default)"
  echo "Chemicals collection:         $(run-mongo-js "db.chemicals.estimatedDocumentCount({})" | tr -d '\r') / 19188359 expected (default)"
  echo "Reactions collection:         $(run-mongo-js "db.reactions.estimatedDocumentCount({})" | tr -d '\r') / 3052316 expected (default, 7072474 including cas)"
  echo "Retro template collection:    $(run-mongo-js "db.retro_templates.estimatedDocumentCount({})" | tr -d '\r') / 432538 expected (default, 612013 including cas)"
  echo "Forward template collection:  $(run-mongo-js "db.forward_templates.estimatedDocumentCount({})" | tr -d '\r') / 17089 expected (default)"
  echo "Sites ref collection:         $(run-mongo-js "db.sites_refs.estimatedDocumentCount({})" | tr -d '\r') / 123 expected (default)"
}

seed-db() {
  if [ -z "$BUYABLES" ] && [ -z "$CHEMICALS" ] && [ -z "$REACTIONS" ] && [ -z "$RETRO_TEMPLATES" ] && [ -z "$FORWARD_TEMPLATES" ] && [ -z "$REFERENCES" ]; then
    echo "Nothing to seed!"
    echo "Example usages:"
    echo "    bash deploy.sh seed-db -r default                  seed only the default retro templates"
    echo "    bash deploy.sh seed-db -r templates.json.gz        seed retro templates from local file templates.json.gz"
    echo "    bash deploy.sh seed-db -x default                  seed only the default reactions"
    echo "    bash deploy.sh seed-db -x reactions.json.gz        seed reactions from local file reactions.json.gz"
    echo "    bash deploy.sh seed-db -c default                  seed only the default chemical (i.e., historian)"
    echo "    bash deploy.sh seed-db -c historian.json.gz        seed chemicals from local file historian.json.gz"
    echo "    bash deploy.sh set-db-defaults seed-db             seed all default collections"
    return
  fi

  echo "Starting the mongo container for seeding db"
  docker compose -f compose.yaml up -d mongo precompute
  sleep 3

  echo "Seeding mongo database..."
  DATA_DIR="/usr/local/askcos-data/db"

  if [ "$BUYABLES" = "default" ]; then
    echo "Loading default buyables data..."
    buyables_file="${DATA_DIR}/buyables/mcule_buyables_fd2.json.gz"
    mongoimport_upsert buyables "$buyables_file"

    buyables_file="${DATA_DIR}/buyables/chembridge_buyables.json.gz"
    DB_DROP="" mongoimport_upsert buyables "$buyables_file"

    buyables_file="${DATA_DIR}/buyables/buyables.json.gz"
    DB_DROP="" mongoimport_upsert buyables "$buyables_file"
  elif [ -f "$BUYABLES" ]; then
    echo "Loading buyables data from $BUYABLES..."
    buyables_file="${DATA_DIR}/buyables/$(basename "$BUYABLES")"
    docker cp "$BUYABLES" "${COMPOSE_PROJECT_NAME}"-mongo-1:"$buyables_file"
    mongoimport_upsert buyables "$buyables_file"
  fi

  if [ "$CHEMICALS" = "default" ]; then
    echo "Loading default chemicals data..."
    chemicals_file="${DATA_DIR}/historian/chemicals.json.gz"
    mongoimport chemicals "$chemicals_file"

    chemicals_file="${DATA_DIR}/historian/historian.pistachio.json.gz"
    DB_DROP="" mongoimport chemicals "$chemicals_file"

    chemicals_file="${DATA_DIR}/historian/historian.bkms_metabolic.json.gz"
    DB_DROP="" mongoimport chemicals "$chemicals_file"
  elif [ "$CHEMICALS" = "pistachio" ]; then
    echo "Loading pistachio chemicals data..."
    chemicals_file="${DATA_DIR}/historian/historian.pistachio.json.gz"
    mongoimport chemicals "$chemicals_file"
  elif [ "$CHEMICALS" = "bkms" ]; then
    echo "Loading bkms chemicals data..."
    chemicals_file="${DATA_DIR}/historian/historian.bkms_metabolic.json.gz"
    mongoimport chemicals "$chemicals_file"
  elif [ -f "$CHEMICALS" ]; then
    echo "Loading chemicals data from $CHEMICALS..."
    chemicals_file="${DATA_DIR}/historian/$(basename "$CHEMICALS")"
    docker cp "$CHEMICALS" "${COMPOSE_PROJECT_NAME}"-mongo-1:"$chemicals_file"
    mongoimport chemicals "$chemicals_file"
  fi

  if [ "$REACTIONS" = "default" ]; then
    echo "Loading default reactions data..."
    reactions_file="${DATA_DIR}/historian/reactions.pistachio.json.gz"
    mongoimport_upsert reactions "$reactions_file"

    reactions_file="${DATA_DIR}/historian/reactions.bkms_metabolic.json.gz"
    DB_DROP="" mongoimport_upsert reactions "$reactions_file"

    reactions_file="${DATA_DIR}/historian/reactions.USPTO_FULL.json.gz"
    DB_DROP="" mongoimport_upsert reactions "$reactions_file"

    reactions_file="${DATA_DIR}/historian/reactions.cas.min.json.gz"
    if [ -f "$reactions_file" ]; then
      DB_DROP="" mongoimport_upsert reactions "$reactions_file"
    fi

    precompute reactions
  elif [ "$REACTIONS" = "pistachio" ]; then
    echo "Loading pistachio reactions data..."
    reactions_file="${DATA_DIR}/historian/reactions.pistachio.json.gz"
    mongoimport_upsert reactions "$reactions_file"
  elif [ "$REACTIONS" = "cas" ]; then
    echo "Loading cas reactions data..."
    reactions_file="${DATA_DIR}/historian/reactions.cas.min.json.gz"
    mongoimport_upsert reactions "$reactions_file"
  elif [ "$REACTIONS" = "bkms" ]; then
    echo "Loading bkms reactions data..."
    reactions_file="${DATA_DIR}/historian/reactions.bkms_metabolic.json.gz"
    mongoimport_upsert reactions "$reactions_file"

    precompute reactions
  elif [ "$REACTIONS" = "USPTO_FULL" ]; then
    echo "Loading USPTO_FULL reactions data..."
    reactions_file="${DATA_DIR}/historian/reactions.USPTO_FULL.json.gz"
    mongoimport_upsert reactions "$reactions_file"

    precompute reactions
  elif [ -f "$REACTIONS" ]; then
    echo "Loading reactions data from $REACTIONS..."
    reactions_file="${DATA_DIR}/historian/$(basename "$REACTIONS")"
    docker cp "$REACTIONS" "${COMPOSE_PROJECT_NAME}"-mongo-1:"$reactions_file"
    mongoimport_upsert reactions "$reactions_file"

    precompute reactions
  fi

  if [ "$RETRO_TEMPLATES" = "default" ]; then
    echo "Loading default retrosynthetic templates..."
    retro_file="${DATA_DIR}/templates/retro.templates.reaxys.json.gz"
    mongoimport retro_templates "$retro_file"

    retro_file="${DATA_DIR}/templates/retro.templates.pistachio.json.gz"
    DB_DROP="" mongoimport retro_templates "$retro_file"

    retro_file="${DATA_DIR}/templates/retro.templates.bkms_metabolic.json.gz"
    DB_DROP="" mongoimport retro_templates "$retro_file"

    retro_file="${DATA_DIR}/templates/retro.templates.pistachio_ringbreaker.json.gz"
    DB_DROP="" mongoimport retro_templates "$retro_file"

    retro_file="${DATA_DIR}/templates/retro.templates.reaxys_biocatalysis.json.gz"
    DB_DROP="" mongoimport retro_templates "$retro_file"

    retro_file="${DATA_DIR}/templates/retro.templates.cas.json.gz"
    if [ -f "retro_file" ]; then
      DB_DROP="" mongoimport retro_templates "$retro_file"
    fi
  elif [ "$RETRO_TEMPLATES" = "pistachio" ]; then
    echo "Loading pistachio retrosynthetic templates..."
    retro_file="${DATA_DIR}/templates/retro.templates.pistachio.json.gz"
    mongoimport retro_templates "$retro_file"
  elif [ "$RETRO_TEMPLATES" = "cas" ]; then
    echo "Loading cas retrosynthetic templates..."
    retro_file="${DATA_DIR}/templates/retro.templates.cas.json.gz"
    mongoimport retro_templates "$retro_file"
  elif [ "$RETRO_TEMPLATES" = "bkms" ]; then
    echo "Loading bkms retrosynthetic templates..."
    retro_file="${DATA_DIR}/templates/retro.templates.bkms_metabolic.json.gz"
    mongoimport retro_templates "$retro_file"
  elif [ "$RETRO_TEMPLATES" = "ringbreaker" ]; then
    echo "Loading ringbreaker retrosynthetic templates..."
    retro_file="${DATA_DIR}/templates/retro.templates.pistachio_ringbreaker.json.gz"
    mongoimport retro_templates "$retro_file"
  elif [ -f "$RETRO_TEMPLATES" ]; then
    echo "Loading retrosynthetic templates from $RETRO_TEMPLATES ..."
    retro_file="${DATA_DIR}/templates/$(basename "$RETRO_TEMPLATES")"
    docker cp "$RETRO_TEMPLATES" "${COMPOSE_PROJECT_NAME}"-mongo-1:"$retro_file"
    mongoimport retro_templates "$retro_file"
  fi

  if [ "$FORWARD_TEMPLATES" = "default" ]; then
    echo "Loading default forward templates..."
    forward_file="${DATA_DIR}/templates/forward.templates.json.gz"
    mongoimport forward_templates "$forward_file"
  elif [ -f "$FORWARD_TEMPLATES" ]; then
    echo "Loading forward templates from $FORWARD_TEMPLATES ..."
    forward_file="${DATA_DIR}/templates/$(basename "$FORWARD_TEMPLATES")"
    docker cp "$FORWARD_TEMPLATES" "${COMPOSE_PROJECT_NAME}"-mongo-1:"$forward_file"
    mongoimport forward_templates "$forward_file"
  fi

  if [ "$SELEC_REFERENCES" = "default" ]; then
    echo "Loading default model references..."
    refs_file="${DATA_DIR}/references/site_selectivity.refs.json.gz"
    mongoimport sites_refs "$refs_file"
  fi

  index-db
  echo "Seeding db completed."
  count-mongo-docs
  docker compose -f compose.yaml rm -sf mongo precompute
  echo
}

generate-deployment-scripts() {
  python scripts/pre_deploy.py
}

get-images() {
  echo "Getting images using the script from the latest deployment directory"
  echo "    bash deployment/deployment_latest/get_images.sh"
  bash deployment/deployment_latest/get_images.sh
  echo "Images ready."
}

download-db-data() {
  echo "Downloading data for db using scripts/download_data.sh"
  bash scripts/download_data.sh
  echo "Data downloaded."
}

start-services() {
  echo "Start services using the script from the latest deployment directory"
  echo "    bash deployment/deployment_latest/start_services.sh"
  bash deployment/deployment_latest/start_services.sh
  echo "Services started."
}

stop-services() {
  echo "Stop services using the script from the latest deployment directory"
  echo "    bash deployment/deployment_latest/stop_services.sh"
  bash deployment/deployment_latest/stop_services.sh
  echo "Services stopped."
}

remove-volumes() {
  docker compose -f compose.yaml down -v
}

export_volume() {
  volume=$1
  directory=$2
  filename=$3
  volume_name=${COMPOSE_PROJECT_NAME}_${volume}
  docker run --rm -v "${volume_name}":/src -v "${directory}":/dest alpine tar -czf /dest/"${filename}" /src
}

import_volume() {
  volume=$1
  directory=$2
  filename=$3
  volume_name=${COMPOSE_PROJECT_NAME}_${volume}
  docker run --rm -v "${volume_name}":/dest -v "${directory}":/src alpine tar -xzf /src/"${filename}" -C /dest --strip 1
  docker run --rm -it -v deploy_askcosv2_mongo_data:/dest -v /home/ubuntu:/src alpine
  tar -xzf /src/"${filename}" -C /dest --strip 1
}

backup() {
  if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$(pwd)/backup/$(date +%Y%m%d%s)"
  fi
  mkdir -p "${BACKUP_DIR}"
  echo "Backing up data to ${BACKUP_DIR}"
  echo "This may take a few minutes..."
  export_volume askcosv2_mongo_data "${BACKUP_DIR}" mongo_data.tar.gz
  echo "Backup complete."
}

restore() {
  if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$(pwd)/backup/$(ls -t backup | head -1)"
  fi
  echo "Restoring data from ${BACKUP_DIR}"
  echo "This may take a few minutes..."
  import_volume askcosv2_mongo_data "${BACKUP_DIR}" mongo_data.tar.gz
  echo "Restore complete."
}

mongodump() {
  if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$(pwd)/backup/$(date +%Y%m%d%s)"
  fi
  mkdir -p "${BACKUP_DIR}"
  echo "Dumping mongo database contents to ${BACKUP_DIR}"
  echo "This may take a few minutes..."
  docker compose -f compose.yaml exec -T mongo \
    bash -c 'mongodump --host=${MONGO_HOST} --username=${MONGO_USER} --password=${MONGO_PW} --authenticationDatabase=admin --archive=/data/mongo_dump.gz --gzip'
  docker cp "${COMPOSE_PROJECT_NAME}"-mongo-1:/data/mongo_dump.gz "$BACKUP_DIR"
  echo "Dumping complete."
}

mongorestore() {
  if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$(pwd)/backup/$(ls -t backup | head -1)"
  fi
  echo "Restoring data from ${BACKUP_DIR}"
  echo "This may take a few minutes..."
  echo "Starting the mongo container for seeding db"
  docker compose -f compose.yaml up -d mongo
  sleep 3
  docker cp "$BACKUP_DIR"/mongo_dump.gz "${COMPOSE_PROJECT_NAME}"-mongo-1:/data/mongo_dump.gz
  docker compose -f compose.yaml exec -T mongo \
    bash -c 'mongorestore --host=${MONGO_HOST} --username=${MONGO_USER} --password=${MONGO_PW} --authenticationDatabase=admin --gzip --archive=/data/mongo_dump.gz'
  echo "Restore complete."
  docker compose -f compose.yaml rm -sf mongo
}

mongodump-result-only() {
  if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$(pwd)/backup/$(date +%Y%m%d%s)"
  fi
  mkdir -p "${BACKUP_DIR}"
  echo "Dumping mongo database contents (results only) to ${BACKUP_DIR}"
  echo "This may take a few minutes..."
  docker compose -f compose.yaml exec -T mongo \
    bash -c 'mongodump --host=${MONGO_HOST} --username=${MONGO_USER} --password=${MONGO_PW} --authenticationDatabase=admin --db=results --collection=results --archive=/data/mongo_dump_results.gz --gzip'
  docker cp "${COMPOSE_PROJECT_NAME}"-mongo-1:/data/mongo_dump_results.gz "$BACKUP_DIR"
  echo "Dumping complete."
}

mongodump-user-only() {
  if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$(pwd)/backup/$(date +%Y%m%d%s)"
  fi
  mkdir -p "${BACKUP_DIR}"
  echo "Dumping mongo database contents (users only) to ${BACKUP_DIR}"
  echo "This may take a few minutes..."
  docker compose -f compose.yaml exec -T mongo \
    bash -c 'mongodump --host=${MONGO_HOST} --username=${MONGO_USER} --password=${MONGO_PW} --authenticationDatabase=admin --db=askcos --collection=users --archive=/data/mongo_dump_users.gz --gzip'
  docker cp "${COMPOSE_PROJECT_NAME}"-mongo-1:/data/mongo_dump_users.gz "$BACKUP_DIR"
  echo "Dumping complete."
}

mongorestore-result-only() {
  if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$(pwd)/backup/$(ls -t backup | head -1)"
  fi
  echo "Restoring data from ${BACKUP_DIR}"
  echo "This may take a few minutes..."
  echo "Starting the mongo container for seeding db"
  docker compose -f compose.yaml up -d mongo
  sleep 3
  docker cp "$BACKUP_DIR"/mongo_dump_results.gz "${COMPOSE_PROJECT_NAME}"-mongo-1:/data/mongo_dump.gz
  docker compose -f compose.yaml exec -T mongo \
    bash -c 'mongorestore --host=${MONGO_HOST} --username=${MONGO_USER} --password=${MONGO_PW} --authenticationDatabase=admin --gzip --archive=/data/mongo_dump.gz'
  echo "Restore complete."
  docker compose -f compose.yaml rm -sf mongo
}

mongorestore-user-only() {
  if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$(pwd)/backup/$(ls -t backup | head -1)"
  fi
  echo "Restoring data from ${BACKUP_DIR}"
  echo "This may take a few minutes..."
  echo "Starting the mongo container for seeding db"
  docker compose -f compose.yaml up -d mongo
  sleep 3
  docker cp "$BACKUP_DIR"/mongo_dump_users.gz "${COMPOSE_PROJECT_NAME}"-mongo-1:/data/mongo_dump.gz
  docker compose -f compose.yaml exec -T mongo \
    bash -c 'mongorestore --host=${MONGO_HOST} --username=${MONGO_USER} --password=${MONGO_PW} --authenticationDatabase=admin --gzip --archive=/data/mongo_dump.gz'
  echo "Restore complete."
  docker compose -f compose.yaml rm -sf mongo
}

post-update-message() {
  echo
  echo -e "\033[92m================================================================================\033[00m"
  echo
  echo "The local ASKCOS deployment has been updated to version ${VERSION_NUMBER}!"
  echo
  echo "                      ~~~ Thank you for using ASKCOS! ~~~"
  echo
  echo -e "\033[92m================================================================================\033[00m"
}

# Handle positional arguments, which should be commands
if [ $# -eq 0 ]; then
  # No arguments
  echo "Must provide a valid task, e.g. deploy|update."
  echo "See 'deploy.sh help' for more options."
  exit 1;
else
  for arg in "$@"
  do
    case "$arg" in
      clean-data | start-db-services | save-db | seed-db | copy-nginx-conf | pull-images | generate-deployment-scripts | \
      start-web-services | start-ml-servers | start-celery-workers | set-db-defaults | count-mongo-docs | \
      backup | restore | mongodump | mongorestore | mongodump-result-only | mongodump-user-only | mongorestore-result-only | mongorestore-user-only | \
      index-db | diff-env | post-update-message | old-messages | get-images | download-db-data)
        # This is a defined function, so execute it
        $arg
        ;;
      pre-deploy)
        copy-nginx-conf
        diff-env
        generate-deployment-scripts
        get-images
        download-db-data
        set-db-defaults
        seed-db
        ;;
      deploy)
        # Normal first deployment, do everything (pre-deploy + start-backend-services)
        copy-nginx-conf
        diff-env
        generate-deployment-scripts
        get-images
        download-db-data
        set-db-defaults
        seed-db
        start-services
        ;;
      update)
        # Update an existing configuration, database seeding is not performed
        copy-nginx-conf
        diff-env
        generate-deployment-scripts
        get-images
        start-services
        post-update-message
        ;;
      start)
        # (Re)start existing deployment
        start-services
        ;;
      stop)
        # Stop and remove currently running containers
        stop-services
        ;;
      clean)
        # Clean up current deployment including all data volumes
        echo "This will stop and remove all containers and also remove all data volumes, including user data."
        echo "Are you sure you want to continue?"
        read -rp 'Continue (y/N): ' response
        case "$response" in
          [Yy] | [Yy][Ee][Ss])
            echo "Cleaning deployment."
            stop-services
            remove-volumes
            ;;
          *)
            echo "Doing nothing."
            ;;
        esac
        ;;
      restart)
        # Stop and remove currently running containers before starting
        stop-services
        start-services
        ;;
      *)
        echo "Error: Unsupported command $arg" >&2  # print to stderr
        exit 1;
    esac
  done
fi
