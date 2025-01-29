# this script first destroys the supernode, if already running,
# and then restarts it
kill -9 "$(ps aux | grep '[f]lower-supernode' | awk '{print $2}')"

nohup poetry run flower-supernode fedl \
      --server $1:9092 \
      --insecure \
      --node-config "partition-id=$2" \
      > nohup-supernode.out &
