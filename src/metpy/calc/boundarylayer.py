# Copyright (c) 2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains calculations of planetary boundary layer height estimation

Adaptation of previous codes from Diane Tzanos and Thomas Rieutord

Different methods are implemented : 
- Parcel method: intersection of potential temperature profile with its foot value [Sei00, HL06, SD10, Col14]
- maximum vertical gradient of potential and virtual potential temperature [Stull88, HL06, SD10]
- minimum of gradient of specific or relative humidity [Sei00, HL06, Col14]
- Bulk Richardson Number [Stull88, VH96, Sei00, Col14, Guo16]
- Surface-based inversion [SD10, Col14]

References:
-----------
[Stull88]: Stull, R. B. (1988):
    An Introduction to Boundary Layer Meteorology. 
    Vol. 13, Kluwer Academic Publishers.
 
[VH96]: Vogelezang, D. H. P., & Holtslag, A. A. M. (1996):
    Evaluation and model impacts of alternative boundary-layer height formulations.
    Boundary-Layer Meteorology, 81(3-4), 245-269.
    
[Sei00]: Seibert, Petra et al. (2000):
    Review and intercomparison of operational methods for the determination of the mixing height.
    Atmospheric Environment 34, 1001-1027.

[HL06]: Hennemuth, B., & Lammert, A. (2006):
    Determination of the atmospheric boundary layer height from radiosonde and lidar backscatter.
    Boundary-Layer Meteorology, 120(1), 181-200.

[SD10]: Seidel, D. J., Ao, C. O., & Li, K. (2010):
    Estimating climatological planetary boundary layer heights from radiosonde observations: Comparison of methods and uncertainty analysis.
    Journal of Geophysical Research: Atmospheres, 115(D16).   
    
[Col14]: Collaud Coen, M., Praz, C., Haefele, A., Ruffieux, D., Kaufmann, P., and Calpini, B. (2014): 
    Determination and climatology of the planetary boundary layer height above the Swiss plateau by in situ and remote sensing measurements as well as by the COSMO-2 model
    Atmos. Chem. Phys., 14, 13205–13221.
    
[Guo16]: Guo, J., Miao, Y., Zhang, Y., Liu, H., Li, Z., Zhang, W., ... & Zhai, P. (2016):
    The climatology of planetary boundary layer height in China derived from radiosonde and reanalysis data.
    Atmos. Chem. Phys, 16(20), 13309-13319.
"""

import numpy as np

#from .tools import first_derivative
#from ..units import units
#from .. import constants as mpconsts

from metpy.calc import first_derivative
from metpy.units import units
import metpy.constants as mpconsts


def bulk_richardson_number(height, potential_temperature, u, v, b=100, ustar=0 * units('m/s'), vertical_dim=0):
    """Calculate the bulk Richardson number.
    
    See [VH96], eq. (3):
    .. math::   Ri = (g/\theta) * \frac{(\Delta z)(\Delta \theta)}
             {\left(\Delta u)^2 + (\Delta v)^2 + b(u_*)^2}
    
    Parameters
    ----------
    height : `pint.Quantity`
            Atmospheric height
            
    potential_temperature : `pint.Quantity`
            Atmospheric potential temperature
            
    u : `pint.Quantity`
        Profile of wind component in the X (East-West) direction
            
    v : `pint.Quantity`
        Profile of wind component in the Y (North-South) direction
    
    b: float, optional
        Coefficient, defaults to 100 based on [VH96]
    
    ustar : `pint.Quantity`
        Friction velocity
                
    vertical_dim : int, optional
        The axis corresponding to vertical, defaults to 0. Automatically determined from
        xarray DataArray arguments.  
        
    Returns
    -------
    `pint.Quantity`
        Bulk Richardson number
    """
       
    u[0]=0 * units.meter_per_second
    v[0]=0 * units.meter_per_second
    
    Dtheta = potential_temperature-potential_temperature[0]

    Du = u-u[0]
    Dv = v-v[0]
    Dz = height-height[0]
    
    return (mpconsts.g / potential_temperature[0]) * (Dtheta * Dz)/ (Du ** 2 + Dv ** 2)

def boundary_layer_height_from_parcel(height,potential_temperature):
    """Calculate atmospheric boundary layer height with the parcel method.
    
    Parameters
    ----------
    height : `pint.Quantity`
            Atmospheric height profile
            
    potential_temperature : `pint.Quantity`
            Atmospheric potential temperature profile
    
    Returns
    -------
    `pint.Quantity`
        Boundary layer height
        
    Notes
    -----        
    It is the height where the potential temperature profile reaches its
    foot value. It is well suited for unstable conditions.
    See [Sei00, HL06, SD10, Col14].
    
    """
     
    if len(np.where(potential_temperature>potential_temperature[0])[0])>0:
        boundary_layer_height=height[np.where(potential_temperature>potential_temperature[0])[0][0]]
    else:
        boundary_layer_height=np.nan * units.meter
    
    return boundary_layer_height

def boundary_layer_height_from_temperature(height,temperature):
    """Calculate atmospheric boundary layer height from the 
    temperature gradient
        
    Parameters
    ----------
    height : `pint.Quantity`
            Atmospheric height profile
            
    temperature : `pint.Quantity`
            Atmospheric temperature profile
    
    Returns
    -------
    `pint.Quantity`
        Boundary layer height

    Notes
    -----          
    It is the height where the temperature gradient changes of sign.
    It can be used for stable or unstable conditions. See [Col14].   
    
    """
    
    dTdz = first_derivative(temperature,x=height)   
    
    if len(np.where(dTdz*dTdz[0]<0)[0])>0:
        boundary_layer_height=height[np.where(dTdz*dTdz[0]<0)[0][0]]
    else:
        boundary_layer_height=np.nan * units.meter
        
    return boundary_layer_height


def boundary_layer_height_from_bulk_richardson_number(height, bulk_richardson_number, bri_threshold=0.25 * units.dimensionless):
    """Calculate atmospheric boundary layer height with the
    bulk Richardson number method.
    
    Parameters
    ----------
    height : `pint.Quantity`
            Atmospheric height profile
            
    bulk_richardson_number : `pint.Quantity`
            Atmospheric bulk richardson number profile
            
    bri_threshold : `pint.Quantity`, optional
            Threshold to exceed to get boundary layer top. Defaults to 0.25. 
    
    Returns
    -------
    `pint.Quantity`
        Boundary layer height
    
    Notes
    -----
    It is the height where the bulk Richardson number exceeds a given threshold.
    It is well suited for unstable conditions. See [Stull88, VH96, Sei00, Col14, Guo16]. 
    
    """
        
    z = height[~np.isnan(bulk_richardson_number)]
    bulk_richardson_number = bulk_richardson_number[~np.isnan(bulk_richardson_number)]

    if len(np.where(bulk_richardson_number>bri_threshold)[0])>0:
        iblh=np.where(bulk_richardson_number>bri_threshold)[0][0]
        blh=z[iblh]
    else:
        blh=np.nan * units.meter

    return blh


def boundary_layer_height_from_relative_humidity(height,relative_humidity): 
    """Calculate atmospheric boundary layer height from the relative
    humidity gradient
    
    Parameters
    ----------
    height : `pint.Quantity`
            Atmospheric height profile
            
    relative_humidity : `pint.Quantity`
            Atmospheric relative humidity profile
    
    Returns
    -------
    `pint.Quantity`
        Boundary layer height
        
    Notes
    -----        
    It is the height where the relative humidity gradient reaches a minimum.
    See [Sei00, HL06, Col14].
    
    """

    dRHdz = first_derivative(relative_humidity,x=height)    
    return height[np.argmin(dRHdz)]



def boundary_layer_height_from_specific_humidity(height,specific_humidity):
    """Calculate atmospheric boundary layer height from the specific
    humidity gradient
    
    Parameters
    ----------
    height : `pint.Quantity`
            Atmospheric height profile
            
    specific_humidity : `pint.Quantity`
            Atmospheric relative humidity profile
    
    Returns
    -------
    `pint.Quantity`
        Boundary layer height
        
    Notes
    -----          
    It is the height where the specific humidity gradient reaches a minimum.
    See [Sei00, HL06, Col14].        

    """
    
    dqdz = first_derivative(specific_humidity,x=height)    
    return height[np.argmin(dqdz)]



def boundary_layer_height_from_potential_temperature(height,potential_temperature):
    """Calculate atmospheric boundary layer height from the
    potential temperature gradient

    
    Parameters
    ----------
    height : `pint.Quantity`
            Atmospheric height profile
            
    potential_temperature : `pint.Quantity`
            Atmospheric potential temperature profile
    
    Returns
    -------
    `pint.Quantity`
        Boundary layer height

    Notes
    -----          
    It is the height where the potential temperature gradient reaches a maximum.
    See [Stull88, HL06, SD10].
    
    """

    dthetadz = first_derivative(potential_temperature,x=height)   
    return height[np.argmax(dthetadz)]    


def boundary_layer_height_from_virtual_potential_temperature(height,virtual_potential_temperature):
    """Calculate atmospheric boundary layer height from the virtual
    potential temperature gradient

    
    Parameters
    ----------
    height : `pint.Quantity`
            Atmospheric height profile
            
    virtual_potential_temperature : `pint.Quantity`
            Atmospheric virtual potential temperature profile
    
    Returns
    -------
    `pint.Quantity`
        Boundary layer height

    Notes
    -----          
    It is the height where the virtual potential temperature gradient reaches a maximum.
    See [Stull88, HL06, SD10].
    
    """

    dthetavdz = first_derivative(virtual_potential_temperature,x=height)
    return height[np.argmax(dthetavdz)]