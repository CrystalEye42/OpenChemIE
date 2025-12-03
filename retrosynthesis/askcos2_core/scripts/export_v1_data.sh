#!/bin/bash
# This script should be able to run from anywhere, on the same machine as the V1 deployment machine

# Change these variables to the actual name
export V1_APP_CONTAINER_NAME=askcos_app_1
export V1_MONGO_CONTAINER_NAME=askcos_mongo_1
export MYSQL_DUMP_PATH_ON_HOST=/home/ztu/mysql_dump.json
export MONGO_DUMP_PATH_ON_HOST=/home/ztu/mongo_dump.json

# Shouldn't required changes to the following
export DJANGO_SETTINGS_MODULE=askcos_site.settings
export MYSQL_DUMP_PATH_IN_CONTAINER=/home/mysql_dump.json
export MONGO_DUMP_PATH_IN_CONTAINER=/home/mongo_dump.json
export MONGO_HOST=mongo
export MONGO_USER=askcos
export MONGO_PW=askcos

echo "Dumping data in MySQL within the container $V1_APP_CONTAINER_NAME.."
echo "The WARNINGS of (?: (urls.W005) URL namespace 'v2' isn't unique.) can be safely ignored."
docker exec \
  -u root \
  -e DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE \
  $V1_APP_CONTAINER_NAME \
  django-admin dumpdata \
    auth.user main.blacklistedchemicals main.blacklistedreactions main.savedresults \
    --indent 4 \
    -o $MYSQL_DUMP_PATH_IN_CONTAINER
echo "Dumped."

echo "Copying from $V1_APP_CONTAINER_NAME to $MYSQL_DUMP_PATH_ON_HOST on the host machine"
docker cp \
  $V1_APP_CONTAINER_NAME:$MYSQL_DUMP_PATH_IN_CONTAINER \
  $MYSQL_DUMP_PATH_ON_HOST
echo "Copied."

echo "Dumping results data in Mongo within the container $V1_MONGO_CONTAINER_NAME.."
docker exec \
  -u root \
  $V1_MONGO_CONTAINER_NAME \
  mongoexport \
    --host "${MONGO_HOST}" \
    --username "${MONGO_USER}" \
    --password "${MONGO_PW}" \
    --authenticationDatabase admin \
    --db results \
    --collection results \
    --type json \
    --jsonArray \
    -o $MONGO_DUMP_PATH_IN_CONTAINER
echo "Dumped."

echo "Copying from $V1_MONGO_CONTAINER_NAME to $MONGO_DUMP_PATH_ON_HOST on the host machine"
docker cp \
  $V1_MONGO_CONTAINER_NAME:$MONGO_DUMP_PATH_IN_CONTAINER \
  $MONGO_DUMP_PATH_ON_HOST
echo "Copied."
