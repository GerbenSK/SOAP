#!/bin/bash
#
# This runs SOAP on a FLAMINGO simulation box of choice
#
set -e

for sim_name in L100_m9_JETS_frac_0.1 L100_m9_JETS_frac_0.2 L100_m9_JETS_frac_0.5
do
    for redshift in 48 58 68
    do
        echo "============================================================="
        echo " Processing simulation: $sim_name at file $redshift"
        echo "============================================================="

        # Run the group membership script
        mpirun -np 8 python -u SOAP/group_membership.py \
            --sim-name="$sim_name" \
            --snap-nr=$redshift \
            parameter_files/FLAMINGO_Gerben.yml

        # Calculate halo properties
        mpirun -np 16 python -u SOAP/compute_halo_properties.py \
            --sim-name="$sim_name" \
            --snap-nr=$redshift \
            --chunks=1 \
            parameter_files/FLAMINGO_Gerben.yml

        echo "Finished processing: $sim_name at file $redshift"
    done
done

echo "All simulations completed!"