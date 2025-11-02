#!/bin/bash
#
# Script to monitor Ray job status on Kubernetes
#
# Usage:
#   ./kubernetes/monitor-job.sh

JOB_NAME="hide-and-seek-training"

echo "========================================="
echo "Monitoring Ray Job: $JOB_NAME"
echo "========================================="
echo ""

# Check if job exists
if ! kubectl get rayjob $JOB_NAME &> /dev/null; then
    echo "Error: RayJob '$JOB_NAME' not found"
    exit 1
fi

# Show job status
echo "Job Status:"
kubectl get rayjob $JOB_NAME
echo ""

# Show pods
echo "Pods:"
kubectl get pods -l ray.io/cluster=$JOB_NAME
echo ""

# Show recent events
echo "Recent Events:"
kubectl get events --field-selector involvedObject.name=$JOB_NAME --sort-by='.lastTimestamp' | tail -10
echo ""

# Tail logs from head node
echo "========================================="
echo "Tailing logs from head node..."
echo "Press Ctrl+C to stop"
echo "========================================="
echo ""

HEAD_POD=$(kubectl get pods -l ray.io/cluster=$JOB_NAME,ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

if [ -n "$HEAD_POD" ]; then
    kubectl logs -f $HEAD_POD
else
    echo "Head pod not found or not ready yet"
fi
