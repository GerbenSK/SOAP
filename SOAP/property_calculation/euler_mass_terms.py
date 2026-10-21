#! /usr/bin/env python

"""
euler_mass_terms.py

Some utility functions to compute SPH-smoothed quantities, which have been
implemented to compute Euler mass terms.

We put them in a separate file because a lot of functions are needed.

To run, we need numba and healpy installed.
"""

import unyt as u
import numpy as np
from math import sqrt
from numba import jit
from numba.typed import List

from SOAP.property_calculation.spherical_coordinates import project_to_spherical_components

#===============================================================================
# Euler mass integrands
#===============================================================================

def get_integrand_thermal_support(dP_dr, rho):
    return (-4 * np.pi * u.G)**(-1) * dP_dr * rho**(-1)

def get_integrand_turbulent_support(dTurbulence_dr, rho):
    return  (-4 * np.pi * u.G)**(-1) * dTurbulence_dr * rho**(-1)

def get_integrand_rotational_support(vtheta,vphi, R):
    return  (4 * np.pi * u.G)**(-1) * (vtheta**2 + vphi**2) / R

def get_integrand_streaming_support(vr, vtheta, vphi, dvr_dr, dvr_dtheta_R, dvr_dphi_R_sin_theta):
    return  (-4 * np.pi*  u.G)**(-1) * ( (vr * dvr_dr) + (vtheta * dvr_dtheta_R) + (vphi * dvr_dphi_R_sin_theta) )

def do_surface_integral(integrand, surface_element):
    return np.nansum(integrand * surface_element, axis=1)

#===============================================================================
# Generation of query points around sphere.
#===============================================================================

def get_radial_bins(inner_radius, outer_radius, number_radial_bins, scale='log'):
    """
    Returns radial bins for the specified range and number of bins.

    Parameters
    ----------
    inner_radius: float
        The inner edge of the first radial bin.
    outer_radius: float
        The outer edge of the last radial bin.
    number_radial_bins: int
        The number of radial bins to generate.
    scale: str, opt
        Whether bins should be spaced linearly or logarithmically (base 10). It
        defaults to a logarithmic spacing.

    Returns
    -------
    bin_centres: np.ndarray
        The centres of each radial bin.
    bin_edges: np.ndarray
        The edges of all radial bins.
    """

    if scale not in ["log", "lin"]:
        raise ValueError("Accepted string values for scale are log or lin.")

    if scale == 'log':
        multiplicative_factor = (outer_radius / inner_radius).to_value()**(1.0 / number_radial_bins)

        # Do not use logspace because we otherwise lose the cosmo array.
        bin_edges   = inner_radius * multiplicative_factor**np.arange(number_radial_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif scale == 'lin':
        bin_edges = np.linspace(inner_radius, outer_radius, number_radial_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, bin_edges

def get_angular_bins(number_angular_bins):
    """
    Generates angular bins from a Healpix distribution.

    Parameters
    ----------
    number_angular_bins: int
        The number of Healpix angular bins to generate. The value should be
        (12 * 2 ** power)

    Returns
    -------
    np.ndarray
        Polar and azimuthal angles of the bins.
    np.ndarray
        Cartesian location of the healpix centres for a unit sphere.

    """

    # We import within the function because the rest of SOAP does not use it.
    import healpy as hp # type: ignore 

    # Obtain the power factor of 2 we require to generate the healpix sphere.
    # This also checks whether the requested number of angular bins is possible
    # to obtain.
    nside = hp.npix2nside(number_angular_bins)

    theta, phi = hp.pix2ang(nside, np.arange(number_angular_bins))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.vstack((theta, phi)), np.vstack((x, y, z)).T

#===============================================================================
# Main function
#===============================================================================

def convert1(x, dtype, depth):# found on a github page https://github.com/numba/numba/issues/7727
    if depth==0:
        # We have reached the end of the nesting-depth,
        # so convert x to a Numpy array.
        y = np.asarray(x, dtype=dtype)
    else:
        # Recursively call this function on all elements of x.
        y = [convert1(x_, dtype=dtype, depth=depth-1) for x_ in x]

        # Convert Python list to Numba list.
        y = List(y)

    return y

@jit(nopython=True, fastmath=True)
def compute_smoothed_field_with_gradient(particle_masses, particle_densities, particle_properties, particle_smoothing_lengths, particle_coordinates, neighbors, query_point):
    """
    Compute SPH-interpolated quantities at a query point using neighboring particles.

    This function evaluates a set of scalar quantities at a given point in space. 
    It loops over neighboring particles, whose kernel (M5) overlap with the interpolation point in space.
    
    Parameters
    ----------
    particle_masses: np.ndarray
        Masses of the particles.
    particle_densities: np.ndarray
        Densities of the particles.
    particle_properties : tuple of np.ndarray
        Tuple of 1D arrays, each representing a scalar field (e.g., pressure, velocity components)
        defined on the particles. Each particle_properties[i][j] corresponds to the value of the i-th field on the j-th particle.
    particle_smoothing_lengths : np.ndarray
        Smoothing lengths of the particles.
    particle_coordinates : np.ndarray
        Coordinates of the particles.
    neighbors : List (numba list)
        List of indices of neighboring particles that lie within the search radius (maximum particle smoothing length).
    query_point : np.ndarray
        1D array of shape (3,) representing the 3D Cartesian coordinate of the query point where the SPH quantities
        are to be interpolated.

    Returns
    -------
    smoothed_density : float
        Smoothed density at the query point.
    smoothed_fields : np.ndarray
        1D array of shape (num_properties,) containing the smoothed values of each quantity at the query point.
    smoothed_field_gradients : np.ndarray
        2D array of shape (num_properties, 3) containing the smoothed gradients of each quantity at the query point.
    quantity_prefactor_sum : float
        Sum of the prefactors (m_j / ρ_j) * W(|r - r_j|, h_j) for all neighboring particles, used for normalization.
    gradient_prefactor_sum : np.ndarray
        1D array of shape (3,) containing the sum of the gradient prefactors (m_j / ρ_j) * ∇W(|r - r_j|, h_j) for all neighboring particles, used for normalization.
    position_difference_matrix : np.ndarray
        2D array of shape (3, 3) representing the sum of the outer products of the position differences (r - r_j) weighted by the gradient prefactors, 
        used for computing the smoothed gradients in Cartesian coordinates.
    """

    # Arrays to hold smoothed properties and gradients.
    num_properties = len(particle_properties)
    smoothed_density = np.float64(0.0)
    smoothed_fields = np.zeros(num_properties, dtype=np.float64)
    smoothed_field_gradients = np.zeros((num_properties, 3), dtype=np.float64)
    quantity_prefactor_sum = np.float64(0.0)
    gradient_prefactor_sum = np.zeros(3, dtype=np.float64)

    position_difference_matrix = np.zeros((3, 3), dtype=np.float64)

    for p_idx in neighbors: # Jit does not accept vectorized operations, so we split each axis up. 
        dx, dy, dz = (
            query_point[0] - particle_coordinates[p_idx, 0], 
            query_point[1] - particle_coordinates[p_idx, 1], 
            query_point[2] - particle_coordinates[p_idx, 2]
        )
        radius = sqrt(dx*dx + dy*dy + dz*dz)

        if radius < particle_smoothing_lengths[p_idx]:
            kernel = kernel_M5(radius, particle_smoothing_lengths[p_idx])
            kernel_derivative = np.array([dx, dy, dz]) * kernel_derivative_M5(radius, particle_smoothing_lengths[p_idx]) # Add the factor (x - x_p) afterwards as jit otherwise complains

            quantity_prefactor = (particle_masses[p_idx] / particle_densities[p_idx]) * kernel
            gradient_prefactor = (particle_masses[p_idx] / particle_densities[p_idx]) * kernel_derivative

            # Update saved prefactors and density
            smoothed_density += particle_masses[p_idx] * kernel
            quantity_prefactor_sum += quantity_prefactor
            gradient_prefactor_sum += gradient_prefactor
            position_difference_matrix -= gradient_prefactor[:, None] * np.array([dx, dy, dz])[None, :]

            for property_index in range(num_properties): # Add the contribution of each particle to the smoothed fields and gradient fields
                smoothed_fields[property_index] += quantity_prefactor * particle_properties[property_index][p_idx]
                smoothed_field_gradients[property_index, :] += gradient_prefactor * particle_properties[property_index][p_idx]

    return smoothed_density, smoothed_fields, smoothed_field_gradients, quantity_prefactor_sum, gradient_prefactor_sum, position_difference_matrix


def get_sph_quantities_with_gradient(tree, particle_coordinates, particle_densities, particle_masses, particle_smoothing_lengths, particle_properties, query_points):
    """
    Compute SPH-interpolated quantities and gradient at a query point using
    neighboring particles.

    Parameters
    ----------
    tree : scipy.spatial.cKDTree
        KDTree built from particle coordinates for fast neighbor searching.
    particle_coordinates: np.ndarray
        Coordinates of the gas particles.
    particle_densities: np.ndarray
        Densities of the gas particles.
    particle_masses: np.ndarray
        Masses of the gas particles.
    particle_smoothing_lengths : np.ndarray
        Smoothing lengths of the gas particles.
    particle_properties: tuple of np.ndarray .
        Tuple of 1D arrays, each representing a scalar property (e.g., pressure, 
        a given velocity component, etc.) associated to each gas particle.
        The array particle_properties[i][j] corresponds to the value of the i-th
        field on the j-th particle.
    query_points : np.ndarray
        2D array of shape (N_query, 3,) representing the 3D Cartesian coordinate
        of every query point where the SPH quantities are to be calculated.

    Returns
    -------
    smoothed_density : np.array with shape (N_query,)
        Smoothed density for every point in space
    smoothed_fields : np.array with shape (N_query, N_quantities)
        Smoothed values of each quantity for every point in space
    smoothed_field_gradients_spherical : np.array with shape (N_query, N_quantities, 3)
        Smoothed values of each gradient for each quantity for every point in space, projected to spherical coordinates
    """
    num_query_points, num_properties = len(query_points), len(particle_properties)

    smoothed_density = np.zeros(num_query_points, dtype=np.float64)
    smoothed_fields = np.zeros((num_query_points, num_properties), dtype=np.float64)
    smoothed_field_gradients = np.zeros((num_query_points, num_properties, 3), dtype=np.float64)
    quantity_prefactor_sum = np.zeros(num_query_points, dtype=np.float64)
    gradient_prefactor_sum = np.zeros((num_query_points, 3), dtype=np.float64)
    position_difference_matrix = np.zeros((num_query_points, 3, 3), dtype=np.float64)

    neighbor_indices = tree.query_ball_point(query_points, r=np.max(particle_smoothing_lengths), workers=1) # Includes too many points on purpose, these are filtered later on
    neighbor_indices = convert1(neighbor_indices, dtype=np.int64, depth=1)

    for i, neighbors_query_point in enumerate(neighbor_indices):
            (   
            smoothed_density[i], 
            smoothed_fields[i, :], 
            smoothed_field_gradients[i, :, :], 
            quantity_prefactor_sum[i], 
            gradient_prefactor_sum[i, :],
            position_difference_matrix[i, : ,:]
        ) = compute_smoothed_field_with_gradient(particle_masses, 
                                                 particle_densities, 
                                                 particle_properties, 
                                                 particle_smoothing_lengths, 
                                                 particle_coordinates, 
                                                 neighbors_query_point, 
                                                 query_points[i])

    ### Method 1: Our old method using the quotient rule, which is numerically unstable when the matrix is singular.
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     smoothed_density /= quantity_prefactor_sum
    #     smoothed_fields /= quantity_prefactor_sum[:, None]
    #     # Old method using quotient rule
    #     smoothed_field_gradients_cartesian = (
    #         smoothed_field_gradients - smoothed_fields[:, :, None] * gradient_prefactor_sum[:, None, :]
    #     ) / quantity_prefactor_sum[:, None, None]

    ### Method 2: New method using first order Taylor expansion and SVD to find the inverse of the singular matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed_density /= quantity_prefactor_sum
        smoothed_fields /= quantity_prefactor_sum[:, None]
        difference_term = smoothed_field_gradients - smoothed_fields[:, :, None] * gradient_prefactor_sum[:, None, :]
        smoothed_field_gradients_cartesian = np.einsum('nlm,nqm->nql', np.linalg.pinv(position_difference_matrix), difference_term)

    ### Method 3: SPH swift method, without weightin.
    # smoothed_field_gradients_cartesian = smoothed_field_gradients

    # For any of the three methods, we need to project the gradients to spherical coordinates.
    smoothed_field_gradients_spherical = project_to_spherical_components(smoothed_field_gradients_cartesian, query_points) 

    return smoothed_density, smoothed_fields, smoothed_field_gradients_spherical


# Taken from Dehnen & Aly (2012), for a quartic spline kernel (used in COLIBRE)
kernel_gamma_M5 = 2.018932
kernel_constant_M5 = 15625.0 * 0.31830988618379067154 / 512.0

@jit(nopython=True, fastmath=True)
def kernel_M5(r: float | np.float32, H: float | np.float32) -> float:
    """
    Kernel implementation of a quartic spline.

    Parameters
    ----------
    r : float or np.float32
        Distance from particle.

    H : float or np.float32
        Kernel width (i.e. radius of compact support of kernel).

    Returns
    -------
    kernel : float
        Contribution to interpolation by particle at distance `r`.

    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = max((1 - ratio), 0.0)**4 - 5 * max((3/5 - ratio), 0.0)**4 + 10 * max((1/5 - ratio), 0.0)**4
    kernel *= kernel_constant_M5 * inverse_H**3

    return kernel

@jit(nopython=True, fastmath=True)
def kernel_derivative_M5(r: float | np.float32, H: float | np.float32) -> float:
    """
    Kernel implementation of the derivative of a quartic spline.

    Parameters
    ----------
    coord : float or np.float32
        xi - x; 1D distance from particle to interpolation point
    
    r : float or np.float32
        Distance from particle.

    H : float or np.float32
        Kernel width (i.e. radius of compact support of kernel).

    Returns
    -------
    kernel : float
        Contribution to interpolation by particle at distance `r`.

    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    if r == 0:
        return 0

    kernel = - 4 * max((1 - ratio), 0.0)**3 + 20 * max((3/5 - ratio), 0.0)**3 - 40 * max((1/5 - ratio), 0.0)**3
    kernel *= (kernel_constant_M5 * inverse_H**4 / r)

    return kernel
