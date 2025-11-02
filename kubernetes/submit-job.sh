#!/bin/bash
#
# Script to submit Ray RLlib training job to Kubernetes cluster
#
# Usage:
#   ./kubernetes/submit-job.sh

set -e

echo "========================================="
echo "Submitting Hide and Seek Training Job"
echo "========================================="

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl not found. Please install kubectl first."
    exit 1
fi

# Check cluster connectivity
echo "Checking cluster connectivity..."
kubectl cluster-info || {
    echo "Error: Cannot connect to Kubernetes cluster"
    exit 1
}

# Apply PVC first (if it doesn't exist)
echo "Creating Persistent Volume Claim..."
kubectl apply -f kubernetes/rayjob.yaml

# Wait for PVC to be bound
echo "Waiting for PVC to be ready..."
kubectl wait --for=condition=Bound pvc/ray-outputs-pvc --timeout=60s || {
    echo "Warning: PVC not bound yet. Continuing anyway..."
}

# Submit the RayJob
echo "Submitting RayJob..."
kubectl apply -f kubernetes/rayjob.yaml

echo ""
echo "========================================="
echo "Job submitted successfully!"
echo "========================================="
echo ""
echo "Monitor job status:"
echo "  kubectl get rayjob hide-and-seek-training"
echo ""
echo "View logs:"
echo "  kubectl logs -l ray.io/cluster=hide-and-seek-training -f"
echo ""
echo "Get detailed status:"
echo "  kubectl describe rayjob hide-and-seek-training"
echo ""
echo "Access Ray Dashboard (after pod is running):"
echo "  kubectl port-forward svc/hide-and-seek-training-head-svc 8265:8265"
echo "  Then open: http://localhost:8265"
echo ""
echo "Delete job:"
echo "  kubectl delete rayjob hide-and-seek-training"
echo ""
