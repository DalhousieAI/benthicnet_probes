#!/bin/bash

echo ""
echo "---- Getting open TCP socket -------"
date
echo ""
echo "TCP socket status:"
ss -tlpn
echo ""

# Try to use a deterministic socket that is defined by the job id number.
# We only use ports in the range 49152-65535 (inclusive), which are the
# Dynamic Ports, also known as Private Ports.
# We want to try to use a deterministic method because that will minimise
# collisions between jobs.
JOB_SOCKET=$(( 49152 + ( $SLURM_JOB_ID % 16384 ) ))

if ss -tulpn | grep -q ":$JOB_SOCKET ";
then
    # We were not able to use our deterministic socket address, so we'll get
    # any available socket instead. By opening the connection briefly, the
    # system is less likely to allocate something else to the socket before
    # we start using it for real. https://unix.stackexchange.com/a/132524/
    JOB_SOCKET="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')";
    echo "Will use port $JOB_SOCKET (allocated to us by a random bind)"
else
    echo "Will use port $JOB_SOCKET (derived from the job ID)"
fi

if ss -tulpn | grep -q ":$JOB_SOCKET ";
then
    echo "Port $JOB_SOCKET appears to be in use after all"
    exit
fi
echo "------------------------------------"
