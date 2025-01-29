# this script first destroys the superlink, if already running,
# and then restarts it
kill -9 "$(ps aux | grep '[f]lower-superlink' | awk '{print $2}')"

nohup poetry run flower-superlink \
      --insecure \
      > nohup-superlink.out &
