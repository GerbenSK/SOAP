import numpy as np
import warnings
import swiftsimio as sw # type: ignore
import matplotlib.pyplot as plt
import unyt
import argparse
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, IntegrationWarning

warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=IntegrationWarning)

#==========================
# Define clumpsy functions 
#==========================
def load_data(file, quantity):
    data = sw.load(file)

    if quantity == "mass":
        smoothed_profile = data.spherical_overdensity_200_crit.test_field1 * unyt.Msun * unyt.Mpc ** (-3)
        particle_profile = data.spherical_overdensity_200_crit.test_field4 * unyt.Msun

    elif quantity == "thermal energy":
        smoothed_profile = data.spherical_overdensity_200_crit.test_field2 * unyt.kg * unyt.Mpc ** (-1) * unyt.s ** (-2)
        particle_profile = data.spherical_overdensity_200_crit.test_field5 * unyt.Msun * unyt.Mpc**2 * unyt.Gyr**(-2)

    elif quantity == "turbulent energy":
        smoothed_profile = data.spherical_overdensity_200_crit.test_field3 * unyt.kg * unyt.Mpc ** (-1) * unyt.s ** (-2)
        particle_profile = data.spherical_overdensity_200_crit.test_field6 * unyt.Msun * unyt.Mpc**2 * unyt.Gyr**(-2)

    log_mass_total = np.log10(data.spherical_overdensity_200_crit.total_mass.to(unyt.Msun))
    SOradius = data.spherical_overdensity_200_crit.soradius.to(unyt.Mpc)

    return smoothed_profile, particle_profile, log_mass_total, SOradius


def integrate_profile(smoothed_profile, bin_centers, bin_edges, dr):

    volume_element = (4.0 * np.pi / 3.0) * (
        bin_edges[:, 1:]**3 - bin_edges[:, :-1]**3
    )

    integrated_profile = volume_element * smoothed_profile

    cumulative_profile = np.cumsum(
        integrated_profile, axis=1
    )

    # num_halos, num_bins = smoothed_profile.shape

    # cumulative_profile = np.zeros_like(smoothed_profile)

    # for h in range(num_halos):

    #     # Extract radial coordinates in Mpc
    #     r = bin_centers[h]

    #     # Profile values
    #     P = smoothed_profile[h]

    #     # Add a central point at r=0
    #     r_extended = np.concatenate(([0.0] * r.units, r))
    #     P_extended = np.concatenate(([P[0]] * P.units, P))

    #     # Cubic spline interpolation
    #     spline = CubicSpline(
    #         r_extended,
    #         P_extended,
    #         bc_type='natural'
    #     )

    #     # Integrate from 0 to each bin center
    #     for i in range(num_bins):
    #         cumulative_profile[h, i] = 4 * np.pi * quad(
    #             lambda x: x**2 * spline(x),
    #             0,
    #             r[i],
    #             epsabs=1e-5,
    #             epsrel=1e-4
    #         )[0]

    return cumulative_profile


def select_units(quantity):
    if quantity == "mass":
        return unyt.Msun
    elif quantity == "thermal energy" or quantity == "turbulent energy":
        return unyt.g * unyt.cm**2 * unyt.s**(-2)


def select_label(quantity):
    if quantity == "mass":
        return r"Enclosed gas mass $[\mathrm{M_\odot}]$"
    elif quantity == "thermal energy":
        return r"Thermal energy $[\mathrm{g}\ \mathrm{cm^2}\ \mathrm{s^{-2}}]$"
    elif quantity == "turbulent energy":
        return r"Turbulent energy $[\mathrm{g}\ \mathrm{cm^2}\ \mathrm{s^{-2}}]$"


def create_labels(axes, radial_bins, angular_bins):
    for i, num_radial_bins in enumerate(radial_bins):
        for j, num_angular_bins in enumerate(angular_bins):
            axes_index = int(4 * i + j)

            # Create label for SPH and particle profiles
            axes[axes_index].plot([0,0], [0,0], c="black", ls="-", label="SPH")
            axes[axes_index].plot([0,0], [0,0], c="black", ls="--", label="Particles")
            axes[axes_index].legend(
                frameon=False,
                loc="lower right"
            )

            axes[axes_index].tick_params(axis='both', which='both', right=True, top=True, labelleft=True, labelbottom=True, direction="in")
            
            if selected_quantity == "mass":
                axes[axes_index].text(
                    x=0.01, 
                    y=10**12.2, 
                    s=f"Radial bins: {num_radial_bins}\nAngular bins: {num_angular_bins}"
                )

            elif selected_quantity == "thermal energy":
                axes[axes_index].text(
                    x=0.01, 
                    y=10**60.5, 
                    s=f"Radial bins: {num_radial_bins}\nAngular bins: {num_angular_bins}"
                )

            elif selected_quantity == "turbulent energy":
                axes[axes_index].text(
                    x=0.01, 
                    y=10**59.5, 
                    s=f"Radial bins: {num_radial_bins}\nAngular bins: {num_angular_bins}"
                )

            if j == 0: # Set y-axis label only for first column
                axes[axes_index].set_ylabel(label)
            else: # Otherwise remove y-ticks and labels
                plt.setp(axes[axes_index].get_yticklabels(), visible=False)

            if i == len(radial_bins) - 1: # Set x-axis label only for last row
                axes[axes_index].xaxis.set_ticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
                axes[axes_index].set_xlabel(r"$r\ [R_\mathrm{200c}]$")
            else: # Otherwise remove x-ticks and labels
                plt.setp(axes[axes_index].get_xticklabels(), visible=False)
    return None

#=================
# Parse arguments
#=================
argparser = argparse.ArgumentParser()
argparser.add_argument("simname", help="Provide name of simulation, e.g. L100m7")
argparser.add_argument("snapnum", help="Provide snapshot number of simulation, e.g. 127 for z=0")
argparser.add_argument("selected_quantity", help="Provide quantity to integrate, e.g. mass", choices=["mass", "thermal energy", "turbulent energy"])

args = argparser.parse_args()

simname = args.simname
snapnum = args.snapnum.zfill(4)
selected_quantity = args.selected_quantity
units = select_units(selected_quantity)
label = select_label(selected_quantity)

base_name = "convergence" + "/" + "halo_properties" + "_" + snapnum

#===============
# Set up figure
#===============
radial_bins = [20, 30, 60]
angular_bins = [12, 48, 108, 192]

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(
    nrows=3, 
    ncols=4, 
    hspace=0.05, 
    wspace=0.05
)

# Add figure axes
axes = [
    fig.add_subplot(gs[i, j]) 
    for i in range(3) 
    for j in range(4)
]

inner_radius, outer_radius = 0.01, 1.2

# Initialize plotting labels and colors
colors = [
    (1.0, 0.0, 1.0),
    (0.1, 0.9, 0.9),
    (0.95, 0.7, 0.0)
]
dark_factor = 0.6
dark = [
    tuple(dark_factor * x for x in (1.0, 0.0, 1.0)), 
    tuple(dark_factor * x for x in (0.1, 0.9, 0.9)),
    tuple(dark_factor * x for x in (0.95, 0.7, 0.0))
]

for i, num_radial_bins in enumerate(radial_bins):
    # Create the radial bins
    multiplicative_factor = (outer_radius / inner_radius)**(1.0 / num_radial_bins)
    bin_edges = inner_radius * multiplicative_factor**np.arange(num_radial_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    dr = np.diff(bin_edges)[None, :]

    for j, num_angular_bins in enumerate(angular_bins):
        print(f"i = {i+1} / {len(radial_bins)}, j = {j+1} / {len(angular_bins)}", flush=True)
        axes_index = int(4 * i + j)

        file = base_name + "_" + str(num_radial_bins) + "_" + str(num_angular_bins) + ".hdf5"

        smoothed_profile, particle_profile, log_mass_total, SOradius = load_data(file, selected_quantity)

        MW_mask = (11.9 <= log_mass_total) & (log_mass_total < 12.1)
        group_mask = (12.9 <= log_mass_total) & (log_mass_total < 13.1)
        cluster_mask = (13.5 <= log_mass_total) & (log_mass_total < 14.5)

        # Now we loop over the three object types (MWs, groups and clusters)
        # and plot the resulting median Euler mass terms for each type
        for k, mask in enumerate([MW_mask, group_mask, cluster_mask]):

            integrated_profile = integrate_profile(
                smoothed_profile[mask],
                bin_centers[None, :] * SOradius[mask, None],
                bin_edges[None, :] * SOradius[mask, None],
                dr * SOradius[mask, None]
            )

            axes[axes_index].loglog(bin_centers, np.mean(integrated_profile, axis=0).to(units), lw=2, c=colors[k], ls="-")
            axes[axes_index].loglog(bin_centers, np.mean(particle_profile[mask], axis=0).to(units), lw=2, c=colors[k], ls="--")

create_labels(axes, radial_bins, angular_bins)

fig.suptitle(
    t=f"Integrated SPH and summed particle {selected_quantity} profiles for MW- (purple), group- (blue)\n"
    f"and cluster-mass (orange) halos for {simname} at different radial and angular resolutions",
    y=0.92
)

plt.savefig(
    f'convergence/{simname}_{selected_quantity}_grid.png',
    bbox_inches='tight',
    dpi=300
)
plt.close()