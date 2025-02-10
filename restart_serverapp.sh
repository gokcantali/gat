# this script first destroys the serverapp, if already running,
# and then restarts it
kill -9 "$(ps aux | grep '[f]lower-server-app' | awk '{print $2}')"

nohup poetry run flower-server-app fedl \
      --superlink $1:9091 \
      --insecure \
      >> $1 2>&1 &
