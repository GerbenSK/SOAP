#!/bin/bash
#
# This runs SOAP on a swift sim L100m9 box
#
set -e

# Run the group membership script
mpirun -np 8 python -u SOAP/group_membership.py \
    --sim-name="L1_m9_frac_0.0" \
    --snap-nr=68 \
    parameter_files/MINIMAL_FLAMINGO_Gerben.yml

# Calculate halo properties
mpirun -np 8 python -u SOAP/compute_halo_properties.py \
    --sim-name="L1_m9_frac_0.0" \
    --snap-nr=68 \
    --chunks=1 \
    parameter_files/MINIMAL_FLAMINGO_Gerben.yml