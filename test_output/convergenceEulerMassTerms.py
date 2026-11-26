import numpy as np
import warnings
import swiftsimio as sw # type: ignore
import matplotlib.pyplot as plt
import unyt
import argparse

warnings.filterwarnings(action="ignore", category=RuntimeWarning)

#==========================
# Define clumpsy functions 
#==========================
def load_euler_mass_terms(file, mass_term):
    data = sw.load(file)

    if mass_term == "thermal":
        euler_term = data.spherical_overdensity_200_crit.euler_mass_thermal_support
    elif mass_term == "turbulence" or mass_term == "turbulent":
        euler_term = data.spherical_overdensity_200_crit.euler_mass_turbulent_support
    elif mass_term == "rotational":
        euler_term = data.spherical_overdensity_200_crit.euler_mass_rotational_support
    elif mass_term == "streaming":
        euler_term = data.spherical_overdensity_200_crit.euler_mass_streaming_support
    else:
        raise RuntimeError(
            f"Selected mass term ({mass_term}) not available is not accepted; "
            f"available options are: thermal, tubulence (or turbulent), rotational, streaming."
        )
    log_mass_total = np.log10(data.spherical_overdensity_200_crit.total_mass.to(unyt.Msun))
    SOradius = data.spherical_overdensity_200_crit.soradius.to(unyt.Mpc)

    return euler_term, log_mass_total, SOradius

def create_labels(axes, radial_bins, angular_bins):
    for i, num_radial_bins in enumerate(radial_bins):
        for j, num_angular_bins in enumerate(angular_bins):
            axes_index = int(4 * i + j)

            axes[axes_index].set_ylim(-0.3, 1.3)
            axes[axes_index].tick_params(axis='both', which='both', right=True, top=True, labelleft=True, labelbottom=True, direction="in")
            axes[axes_index].axhline(y=1, c='black', ls=':', zorder=-1)
            axes[axes_index].axhline(y=0, c='black', ls=':', zorder=-1)
            axes[axes_index].text(
                x=0.01, 
                y=1.05, 
                s=f"Radial bins: {num_radial_bins}\nAngular bins: {num_angular_bins}"
            )

            if j == 0: # Set y-axis label only for first column
                axes[axes_index].set_ylabel(rf"${label}$")
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
argparser.add_argument("euler_mass_term", help="Provide mass term to evaluate, e.g. thermal or streaming")
argparser.add_argument("snapnum", help="Provide snapshot number of simulation, e.g. 127 for z=0")
argparser.add_argument(
    "plot_shaded_areas", 
    help="Choose to plot 25th to 75th percentiles, use 0 for False and 1 for True",
)

args = argparser.parse_args()

simname = args.simname
selected_mass_term = args.euler_mass_term
snapnum = args.snapnum.zfill(4)
plot_shaded_regions = bool(int(args.plot_shaded_areas))

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

# Initialize plotting labsels and colors
label = rf'S_\mathrm{{{selected_mass_term}}}'
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
    bin_edges   = inner_radius * multiplicative_factor**np.arange(num_radial_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    for j, num_angular_bins in enumerate(angular_bins):
        axes_index = int(4 * i + j)

        file = base_name + "_" + str(num_radial_bins) + "_" + str(num_angular_bins) + ".hdf5"

        euler_mass_term, log_mass_total, SOradius = load_euler_mass_terms(file, selected_mass_term)

        MW_mask = (11.9 <= log_mass_total) & (log_mass_total < 12.1)
        group_mask = (12.9 <= log_mass_total) & (log_mass_total < 13.1)
        cluster_mask = (13.5 <= log_mass_total) & (log_mass_total < 14.5)

        # Now we loop over the three object types (MWs, groups and clusters)
        # and plot the resulting median Euler mass terms for each type
        for k, mask in enumerate([MW_mask, group_mask, cluster_mask]):
            median_euler_mass_term = np.nanmedian(euler_mass_term[mask], axis=0)

            axes[axes_index].semilogx(bin_centers, median_euler_mass_term, lw=2.5, c=dark[k], ls="-")
            axes[axes_index].semilogx(bin_centers, median_euler_mass_term, lw=2, c=colors[k], ls="-")

            if plot_shaded_regions:
                axes[axes_index].fill_between(
                    bin_centers,
                    np.percentile(euler_mass_term[mask], 25, axis=0),
                    np.percentile(euler_mass_term[mask], 75, axis=0),
                    color=colors[k],
                    alpha=0.3
                )

create_labels(axes, radial_bins, angular_bins)

fig.suptitle(
    t=f"{selected_mass_term.capitalize()} support for MW- (purple), group- (blue) and cluster-mass (orange) halos\n"
    f"for {simname} at different radial and angular resolutions",
    y=0.92
)

plt.savefig(
    f'convergence/{simname}_{selected_mass_term}_support_grid.png',
    bbox_inches='tight',
    dpi=300
)
plt.close()