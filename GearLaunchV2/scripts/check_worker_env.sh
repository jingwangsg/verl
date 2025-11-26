#!/bin/bash

echo "=== Checking worker_1 files ==="
echo "Node: $(hostname)"
echo "Current directory: $(pwd)"
echo ""

echo "=== Checking /etc/workflow_env ==="
if [ -f /etc/workflow_env ]; then
    echo "✅ /etc/workflow_env exists"
    cat /etc/workflow_env
else
    echo "❌ /etc/workflow_env does not exist"
fi
echo ""

echo "=== Checking /etc/osmo_hosts.txt ==="
if [ -f /etc/osmo_hosts.txt ]; then
    echo "✅ /etc/osmo_hosts.txt exists"
    echo "Number of hosts: $(wc -l < /etc/osmo_hosts.txt)"
    echo "First 5 hosts:"
    head -5 /etc/osmo_hosts.txt
else
    echo "❌ /etc/osmo_hosts.txt does not exist"
fi
echo ""

echo "=== Checking /etc/osmo_hosts_mpi.txt ==="
if [ -f /etc/osmo_hosts_mpi.txt ]; then
    echo "✅ /etc/osmo_hosts_mpi.txt exists"
    echo "Number of hosts: $(wc -l < /etc/osmo_hosts_mpi.txt)"
else
    echo "❌ /etc/osmo_hosts_mpi.txt does not exist"
fi
echo ""

echo "=== End of check ==="