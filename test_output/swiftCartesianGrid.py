import swiftsimio as sw # type: ignore
from swiftsimio.visualisation.volume_render import render_gas # type: ignore
import numpy as np
import unyt
import warnings
warnings.filterwarnings(action="ignore", message="RuntimeWarning: invalid value encountered in divide")
import matplotlib.pyplot as plt

debug = False

snapshot_path = "/net/hypernova/data2/COLIBRE/L0050N0752/Thermal/snapshots/colibre_0123/colibre_0123.hdf5"
catalogue_path = f"/net/hypernova/data2/COLIBRE/L0050N0752/Thermal/SOAP-HBT/halo_properties_0123.hdf5"

print("Loading halo catalogue and selecting halo...")

catalogue = sw.load(catalogue_path)
mass = catalogue.spherical_overdensity_200_crit.total_mass
mass_mask = (mass >= 10**13 * unyt.Msun)

halo_radii = catalogue.spherical_overdensity_200_crit.soradius[mass_mask]
halo_centres = catalogue.input_halos.halo_centre[mass_mask]
mass = mass[mass_mask]

halo_centre = halo_centres[0]
halo_radius = halo_radii[0]
mass = mass[0]

if debug:
    print(f"Selected halo of mass ({mass}) at ({halo_centre}), with radius of ({halo_radius})")

# Mask the data
mask = sw.mask(snapshot_path)

load_region = [[halo_centre[0] - halo_radius, halo_centre[0] + halo_radius],
               [halo_centre[1] - halo_radius, halo_centre[1] + halo_radius],
               [halo_centre[2] - halo_radius, halo_centre[2] + halo_radius]]

mask.constrain_spatial(load_region)

data = sw.load(snapshot_path, mask=mask)

# Region for volume render
region = sw.cosmo_array(
    [
        halo_centre[0] - halo_radius,
        halo_centre[0] + halo_radius,
        halo_centre[1] - halo_radius,
        halo_centre[1] + halo_radius,
        halo_centre[2] - halo_radius,
        halo_centre[2] + halo_radius,
    ],
    halo_centre.units,
    comoving=True,
    scale_factor=data.metadata.a,
    scale_exponent=1,
)

print("Creating data sets...")
# First create a volume-weighted pressure dataset
data.gas.volume_weighted_pressure = (data.gas.masses / data.gas.densities) * data.gas.pressures
data.gas.volumes = data.gas.masses / data.gas.densities

if debug:
    print(f"Halo holds ({len(data.gas.volumes)}) particles")

print("Making 3D Cartesian grid of volume weights...")
# Map in Mpc^3
resolution_elements = 128
volume_cube = render_gas(
    data,
    resolution=resolution_elements,
    project="volumes",
    parallel=True,
    periodic=True,
    region=region
)

print("Making 3D Cartesian grid of volume weighted pressures...")
# Map in pressure / mpc^3
pressure_cube = render_gas(
    data,
    resolution=resolution_elements,
    project="volume_weighted_pressure",
    parallel=True,
    periodic=True,
    region=region
)

# Get the volume weighted pressure cube
volume_weighted_pressure_cube = pressure_cube / volume_cube

if debug:
    number_of_empty_cells = np.sum(volume_weighted_pressure_cube == 0)
    print(f"At a resolution of ({resolution_elements}x{resolution_elements}x{resolution_elements}), "
        f"we have {number_of_empty_cells} empty bins (f={number_of_empty_cells/(resolution_elements**3)}).")

x_bins = np.linspace(-halo_radius, halo_radius, resolution_elements+1)
y_bins = np.linspace(-halo_radius, halo_radius, resolution_elements+1)
z_bins = np.linspace(-halo_radius, halo_radius, resolution_elements+1)

x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

cell_radii = np.sqrt(X * X + Y * Y + Z * Z)

voxel_size = 2 * halo_radius / resolution_elements # Voxel size in Mpc
sqrt2_voxel_size = np.sqrt(2) * voxel_size

num_radial_bins = 30
radius_edges = np.logspace(
    np.log10(0.01 * halo_radius.value), 
    np.log10(1 * halo_radius.value),
    num_radial_bins+1
) * halo_radius.units

radius_centers = np.sqrt(radius_edges[:-1] * radius_edges[1:])

averaged_weighted_pressure = np.zeros(num_radial_bins)

for i in range(num_radial_bins):
    radius_mask = (cell_radii >= radius_centers[i] - sqrt2_voxel_size) & ((cell_radii < radius_centers[i] + sqrt2_voxel_size))
    if debug:
        print(f"For radius ({radius_centers[i]:.4f}), we have {np.sum(radius_mask)} cells intersecting the bin...")
    averaged_weighted_pressure[i] = np.nanmean(volume_weighted_pressure_cube[radius_mask])

plt.suptitle(f"Radial bins: ({num_radial_bins}), Resolution elements: ({resolution_elements}), halo mass: ({mass:.2e})")
plt.loglog(radius_centers / halo_radius, averaged_weighted_pressure)
plt.xlabel(r"$r/r_\mathrm{200c}$")
plt.ylabel("Pressure")
plt.savefig("test_swift_cartesian_grid_log_13.png", bbox_inches="tight", dpi=300)
