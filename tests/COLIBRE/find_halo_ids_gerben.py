#!/bin/env python3
#
# Find IDs of halos in a corner of the simulation box
#
# Run with e.g. `python3 find_halo_ids.py L0025N0188/Thermal 127 11 12`
#
import sys
import numpy as np
import h5py
import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)

def find_halo_indices_MW_groups_clusters(sim, snap_nr):
    soap_file = f"/net/hypernova/data2/COLIBRE/{sim}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"
    with h5py.File(soap_file, "r") as f:
        # Load properties
        mass = np.log10(f["SO/200_crit/TotalMass"][...] * 10**10)
        index = f["InputHalos/HaloCatalogueIndex"][()]

        # Mask on halo mass into MW-like, groups and clusters
        MW_mask = (11.9 <= mass) & (mass < 12.1)
        group_mask = (12.9 <= mass) & (mass < 13.1)
        cluster_mask = (13.5 <= mass) & (mass < 14.5)

        total_mask = (MW_mask | group_mask) | cluster_mask
        print(f"We have ({MW_mask.sum()}) MW-like halos, ({group_mask.sum()}) groups and ({cluster_mask.sum()}) clusters.")
        print(f"In total we have ({total_mask.sum()}) halos selected.")

        return index[total_mask]


if __name__ == "__main__":
    sim = sys.argv[1]
    snap_nr = int(sys.argv[2])

    indices = find_halo_indices_MW_groups_clusters(sim, snap_nr)
    # Print one index per line, so that the output can be redirected to a
    # file which can be passed to SOAP with --halo-indices-file
    print(len(indices))
    with open(f"halo_indices_{snap_nr:04d}.txt", "w") as f:
        # f.write(f"# Halo indices to do: all halos with x<{boxsize}, y<{boxsize}, and z<{boxsize} cMpc in snap {snap_nr}\n")
        f.write(f"# Combination of MW, group and cluster-mass halos\n")
        f.write(f"# Generated with `python3 find_halo_ids_gerben.py {sim} {snap_nr}'\n")
        for i in indices:
            f.write(f"{i}\n")
