# adjust the FL configuration parameters before the simulation
sed -i '' -E "s/NUM_ROUNDS = (.+)/NUM_ROUNDS = $1/g" ./fedl/fedl/server_app.py
sed -i '' -E "s/METHOD = \"(.+)\"/METHOD = \"$2\"/g" ./fedl/fedl/server_app.py
sed -i '' -E "s/TRIAL = \"(.+)\"/TRIAL = \"$3\"/g" ./fedl/fedl/server_app.py

# start the FL simulation, assuming that
# the superlink and the server-app are hosted on the same machine
nohup poetry run flower-server-app fedl \
      --superlink 0.0.0.0:9091 \
      --insecure \
      > FL_$1_Rounds_$2_Algorithm_$3_Trial_Results.txt &
