#!/bin/bash
# This script should be able to run from anywhere, on the same machine as the V2 deployment machine

# Change these variables to the actual name
export V2_APP_CONTAINER_NAME=deploy-app-1
export MYSQL_DUMP_PATH_ON_HOST=/home/ztu/ASKCOSv2/mysql_dump.json
export MONGO_DUMP_PATH_ON_HOST=/home/ztu/ASKCOSv2/mongo_dump.json

# Shouldn't required changes to the following
export MYSQL_DUMP_PATH_IN_CONTAINER=/home/mysql_dump.json
export MONGO_DUMP_PATH_IN_CONTAINER=/home/mongo_dump.json

echo "Copying from $MYSQL_DUMP_PATH_ON_HOST to $V2_APP_CONTAINER_NAME"
docker cp \
  $MYSQL_DUMP_PATH_ON_HOST \
  $V2_APP_CONTAINER_NAME:$MYSQL_DUMP_PATH_IN_CONTAINER
echo "Copied."

echo "Copying from $MONGO_DUMP_PATH_ON_HOST to $V2_APP_CONTAINER_NAME"
docker cp \
  $MONGO_DUMP_PATH_ON_HOST \
  $V2_APP_CONTAINER_NAME:$MONGO_DUMP_PATH_IN_CONTAINER
echo "Copied."

echo "Importing V1 data into V2"
docker exec \
  -e MYSQL_DUMP_PATH_IN_CONTAINER=$MYSQL_DUMP_PATH_IN_CONTAINER \
  -e MONGO_DUMP_PATH_IN_CONTAINER=$MONGO_DUMP_PATH_IN_CONTAINER \
  $V2_APP_CONTAINER_NAME \
  /opt/conda/bin/python -m scripts.import_v1_data_to_v2
echo "Imported"
