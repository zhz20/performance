#!/bin/zsh

set -e  # Exit the script on any command failure
BASE_PATH=""
DEPENDS_PATH="$BASE_PATH/depends-0.9.7"

# Define variables

APP_NAME=""
VERSIONS=()
REPO_PATH="$BASE_PATH/apps/$APP_NAME"


# Loop through each version
for VERSION in "${VERSIONS[@]}"; do
  # Change directory to the Git repository
  cd "$REPO_PATH" || { echo "Failed to navigate to repo"; exit 1; }

  echo "Checking out version $VERSION"
  git checkout $VERSION || { echo "Failed to checkout $VERSION"; exit 1; }


  # Run your shell commands here
  echo "Running mvn test for $VERSION"
  # Commands
  export MAVEN_OPTS="-agentpath:libyjpagent.dylib=tracing,monitors,on_exit=snapshot,alloc_each=1,snapshot_name_format=$APP_NAME-$VERSION,dir=$BASE_PATH/dynamic/$APP_NAME"
  mvn -DforkCount=0 test || echo "mvn test failed for $VERSION"

  # Collect static data
  echo "Collecting static data for $VERSION"
  cd $DEPENDS_PATH
  ./depends.sh java --format json --granularity method $REPO_PATH $APP_NAME-$VERSION
  mv $DEPENDS_PATH/$APP_NAME-$VERSION-method.json $BASE_PATH/static/$APP_NAME/$APP_NAME-$VERSION-method-call.json
  
  echo "Exporting dynamic snapshot to CSV for $VERSION"
  # Export snapshot to CSV files
  java -Dexport.method.list.cpu -Dexport.csv -Xmx8g -jar yourkit.jar -export $BASE_PATH/dynamic/$APP_NAME/$APP_NAME-$VERSION-shutdown.snapshot  $BASE_PATH/dynamic/$APP_NAME
  mv $BASE_PATH/dynamic/$APP_NAME/Method-list-CPU.csv $BASE_PATH/dynamic/$APP_NAME/Method-list-CPU-$VERSION.csv
  
  java -Dexport.method.list.alloc -Dexport.csv -Xmx8g -jar yourkit.jar -export $BASE_PATH/dynamic/$APP_NAME/$APP_NAME-$VERSION-shutdown.snapshot  $BASE_PATH/dynamic/$APP_NAME
  mv $BASE_PATH/dynamic/$APP_NAME/Method-list-allocations.csv $BASE_PATH/dynamic/$APP_NAME/Method-list-allocations-$VERSION.csv

  echo "Completed processing for $VERSION"
done