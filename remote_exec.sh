#!/bin/bash

# Remote execution script with proper environment loading
# Usage: ./remote_exec.sh "<command>" server1 [server2 server3 ...]

# Check if at least 2 arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Invalid arguments"
    echo "Usage: $0 \"<command>\" server1 [server2 server3 ...]"
    exit 1
fi

COMMAND="$1"
shift
SERVERS=("$@")

for server in "${SERVERS[@]}"; do
    echo "=== Executing on $server ==="
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$server" "bash --login -c '
        # Load the appropriate shell environment
        if [ -f ~/.bashrc ]; then
            source ~/.bashrc
        fi
        if [ -f ~/.bash_profile ]; then
            source ~/.bash_profile
        fi
        # Change to the directory where the command should be executed
        cd \$HOME/gat || cd /home/ubuntu/gat
        # Execute the command
        $COMMAND
    '" 2>&1 | sed "s/^/$server: /"
    echo "============================"
    echo
done
