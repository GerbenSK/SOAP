#!/bin/env python3
#
# Find IDs of halos in a corner of the simulation box
#
# Run with e.g. `python3 find_halo_ids.py L0025N0188/Thermal 127 11 12`
#
import sys
import numpy as np
import h5py


def find_halo_indices_box(sim, snap_nr, boxsize):
    soap_file = f"/net/hypernova/data2/COLIBRE/{sim}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"
    with h5py.File(soap_file, "r") as f:
        pos = f["InputHalos/HaloCentre"][()]
        mask = np.all(pos < boxsize, axis=1)
        index = f["InputHalos/HaloCatalogueIndex"][()]
        is_central = f["InputHalos/IsCentral"][()]
        nstar = f['BoundSubhalo/NumberOfStarParticles'][:]
        # Diagnostics are written to stderr, so that stdout can be redirected
        # to a file containing only the halo indices
        if np.sum(is_central[mask]) == 0:
            print('No centrals loaded', file=sys.stderr)
        print(f'Max number of stars: {np.max(nstar[mask])}', file=sys.stderr)
        return index[mask]


def find_halo_indices_mass(sim, snap_nr, lower_mass, upper_mass):
    soap_file = f"/net/hypernova/data2/COLIBRE/{sim}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"
    with h5py.File(soap_file, "r") as f:
        mass = np.log10(f["SO/200_crit/TotalMass"][...] * 10**10)
        mask = (lower_mass <= mass) & (mass < upper_mass)
        index = f["InputHalos/HaloCatalogueIndex"][()]
        is_central = f["InputHalos/IsCentral"][()]
        nstar = f['BoundSubhalo/NumberOfStarParticles'][:]
        # Diagnostics are written to stderr, so that stdout can be redirected
        # to a file containing only the halo indices
        if np.sum(is_central[mask]) == 0:
            print('No centrals loaded', file=sys.stderr)
        print(f'Max number of stars: {np.max(nstar[mask])}', file=sys.stderr)
        return index[mask]


if __name__ == "__main__":
    sim = sys.argv[1]
    snap_nr = int(sys.argv[2])
    lower_mass = float(sys.argv[3]) # Lower mass threshold in log10(M200c)
    upper_mass = float(sys.argv[4]) # Upper mass threshold in log10(M200c)

    indices = find_halo_indices_mass(sim, snap_nr, lower_mass, upper_mass)
    # Print one index per line, so that the output can be redirected to a
    # file which can be passed to SOAP with --halo-indices-file
    print(len(indices))
    with open(f"halo_indices_{snap_nr:04d}.txt", "w") as f:
        # f.write(f"# Halo indices to do: all halos with x<{boxsize}, y<{boxsize}, and z<{boxsize} cMpc in snap {snap_nr}\n")
        f.write(f"# Halos with mass in log10(M200c) greater than or equal to {lower_mass} and smaller than {upper_mass}\n")
        f.write(f"# Generated with `python3 find_halo_ids.py {sim} {snap_nr} {lower_mass} {upper_mass}`\n")
        for i in indices:
            f.write(f"{i}\n")
