"""
Auxilary functions for working with persistence diagrams.
Most of these functions should not be called directly.
"""
from __future__ import annotations
import itertools
from operator import itemgetter, attrgetter
import numpy as np


def death_vector(diagram: list, hom_deg: int = 0):
    """ Returns the death vector in degree 0 for the persistence diagram
    """
    if hom_deg != 0:
        raise NotImplementedError("The death vector is not defined for "
                                  "homological degrees greater than zero.")
    pass

def linear_combination(landscapes: list, coeffs: list):
    """ Compute a linear combination of landscapes
    Parameters
    ----------
    landscapes : list of PersistenceLandscape objects
    coeffs : list, optional
    Returns
    -------
    None.
    """
    result = coeffs[0]*landscapes[0]
    for c, L in enumerate(landscapes):
        result += coeffs[c]*L
    return result

def average_landscape(landscapes: list):
    """ Compute the average of a list of landscapes
    """
    pass

# def _lc_grid(landscapes: list, coeffs: list) -> PersistenceLandscapeGrid:
#     """ Compute the linear combination of a list of PersistenceLandscapeGrids """
#     # First snap them to a common grid
#     start = min(landscapes, key=attrgetter('start')).start
#     stop = max(landscapes, key=attrgetter('stop')).stop
#     num_dims = max(landscapes, key=attrgetter('num_dims')).num_dims
#     snapped_list = [PersistenceLandscapeG]

def union_vals(A,B):
    """
    Extends one list to the length of the other by padding with zero lists.
    AAHelper function for summing grid landscapes.
    """
    diff = A.shape[0] - B.shape[0]
    if diff < 0:
        # B has more entries, so pad A
        A = np.pad(A, pad_width=((0,np.abs(diff)), (0,0)))
        return A, B
    elif diff > 0:
        # A has more entries, so pad B
        B = np.pad(B, pad_width=((0,diff),(0,0)))
        return A, B
    else:
        return A, B

def union_crit_pairs(A, B):
    """ Helper function for summing landscapes.
This should handle all the edge cases, like an empty list.
"""
    result_pairs = []
    A.compute_landscape()
    B.compute_landscape()
    # zip functions in landscapes A and B and pad with None
    for a, b in list(itertools.zip_longest(A.critical_pairs, B.critical_pairs)):
        # B had more functions
        if a == None:
            result_pairs.append(b)
        # A had more functions
        elif b == None:
            result_pairs.append(a)
        # A, B > pos_to_slope_interp > sum_slopes > slope_to_pos_interp
        else:
            result_pairs.append(
                slope_to_pos_interp(
                    sum_slopes(
                        pos_to_slope_interp(a),
                        pos_to_slope_interp(b),
                    )
                )
            )
    return result_pairs

def pos_to_slope_interp(l: list) -> list:
    """
Convert positions of critical pairs to (x-value, slope) pairs. Intended for internal use. Inverse function of ￿slope_to_pos_interp￿.
    Result
    ------
    list
        [(xi,mi)]_i for i in len(function in landscape)
    """
    output = []
        # for sequential pairs in landscape function
    for [[x0,y0], [x1,y1]] in zip(l,l[1:]):
        slope = (y1 - y0)/(x1 - x0)
        output.append([x0,slope])
    output.append([l[-1][0],0])
    return output

def slope_to_pos_interp(l: list) -> list:
    """
Convert positions of (x-value, slope) pairs to critical pairs. Intended for internal use. Inverse function of ￿pos_to_slope_interp￿.
    Result
    ------

61
    list
        [(xi, yi)]_i for i in len(function in landscape)
"""
    output = [[l[0][0],0]]
    # for sequential pairs in [(xi,mi)]_i
    for [[x0, m], [x1, _]] in zip(l, l[1:]):
        # uncover y0 and y1 from slope formula
        y0 = output[-1][1]
        y1 = y0 + (x1 - x0)*m
        output.append([x1, y1])
    return output


def sum_slopes(a: list, b: list) -> list:
    """
Sum two piecewise linear functions, each represented as a list
of pairs (xi,mi), where each xi is the x-value of critical pair and mi is the slope. The input should be of the form of the output of the ￿pos_to_slope_interp' function.
    Result
    ------
    list
"""
    result = []
    am, bm = 0, 0  # initialize slopes
    while len(a) > 0 or len(b) > 0:
        if len(a) == 0 or (len(a) > 0 and len(b) > 0 and a[0][0] > b[0][0]):
            # The next critical pair comes from list b.
            bx, bm = b[0]
            # pop b0
            b = b[1:]
            result.append([bx, am + bm])
        elif len(b) == 0 or (len(a) > 0 and len(b) > 0 and a[0][0] < b[0][0]):
            # The next critical pair comes from list a.
            ax, am = a[0]
            # pop a0
            a = a[1:]
            result.append([ax, am + bm])
        else:
            # The x-values of two critical pairs coincide.
            ax, am = a[0]
            bx, bm = b[0]
            # pop a0 and b0
            a, b = a[1:], b[1:]
            result.append([ax, am + bm])
        # reduce trailing zeroes
   # if len(result) > 2 and result[-1][1] == result[2][1]:
   #     result.pop()
    return result

def pairs_snap(pairs, grid):
    """
    Snap an 2D-array of points to values along an 2D-array grid.
    Each point will be snapped to the grid value with the smallest
    city-block distance.
    Parameters
    ---------
    points: 2D-array. Must have same number of columns as grid
    grid: 2D-array. Must have same number of columns as points
    Returns
    -------
    A 2D-array with one row per row of points. Each i-th row will
    correspond to row of grid to which the i-th row of points is closest.
    In case of ties, it will be snapped to the row of grid with the
    smaller index.
    """
    # transpose grid
    grid_3d = np.transpose(grid[:,:,np.newaxis], [2,1,0])
    # axis 1 is x-values of points
    diffs = np.sum(np.abs(grid_3d - pairs[:,:,np.newaxis]), axis=1)
    # argmin returns the indices of the minimum values along an axis
    best = np.argmin(diffs, axis = 1)
    return  grid[best,:]


# def exact_to_grid(pl: PersistenceLandscapeExact) -> PersistenceLandscapeGrid:
#     """
#     Converts a PersistenceLandscapeExact class to a PersistenceLandscapeGrid class.
#     """

# pass

def values_snap(values, grid):
    # transpose values
    values_transpose = values[:, np.newaxis]
    diffs = np.abs(values_transpose - grid)
    best = np.argmin(diffs, axis = 1)
    return grid[best]


def num_skip(n: int):
    """This should make it easy to throw out the first ￿n￿ landscape functions, regardless of grid or exact pl passed.
"""
    pass

#
# def vectorize(l: PersistenceLandscapeExact, num_dims, start: float = None, stop: float = None):
#     """
# Returns a list of interpolated y-values of ￿self.critical_pairs￿ on user specified grid.
#     Parameters
#     ----------
#     start: float, default None
#         start value of grid
#     if start is not inputed, start is assigned to minimum birth value
#     stop: float, default None
#     stop value of grid
# if stop is not inputed, stop is assigned to maximum death value
# num_dims: int, default 500
# number of points starting from ￿start￿ and ending at ￿stop￿
# """
#     l.compute_landscape()
#     # default start and stop value to min/max birth/death value
#     if not start:
#         start = min(l.critical_values,key=itemgetter(0))[0]
#     if not stop:
#         stop = max(l.critical_values, key=itemgetter(0))[0]
#     grid = np.linspace(start, stop, num_dims)
#     result = []
#     # creates sequential pairs of points for each lambda in critical_pairs
#     for l in self.critical_pairs:
#         xs, ys = zip(*l)
#         result.append(np.interp(grid, xs, ys))
#     return result