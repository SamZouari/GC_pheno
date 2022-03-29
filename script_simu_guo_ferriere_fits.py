
import click
import warnings
import matplotlib.pyplot as plt
import gc
import logging
import os, sys
import time
from pathlib import Path

import numpy as np
import astropy.units as u
import astropy.constants as cst
from astropy.utils import lazyproperty
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.table import Table

from scipy.spatial.transform import Rotation as R

from naima.models import PionDecay, InverseCompton, ExponentialCutoffPowerLaw, TableModel
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.utils.integrate import trapz_loglog

from clouds import Cloud_sphere, Cloud_ring, Cloud_ellipsoid, Cloud_cylinder, create_ferriere_clouds


log = logging.getLogger(__name__)

pathres = Path('../2D_maps')


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)
    if not show_warnings:
        warnings.simplefilter("ignore")
        

def integrate_energy_flux(func, energy_min, energy_max, ndecade=100):
    """Integrate 1d function using the log-log trapezoidal rule.

    Internally an oversampling of the energy bins to "ndecade" is used.

    Parameters
    ----------
    func : callable
        Function to integrate.
    energy_min : `~astropy.units.Quantity`
        Integration range minimum
    energy_max : `~astropy.units.Quantity`
        Integration range minimum
    ndecade : int, optional
        Number of grid points per decade used for the integration.
        Default : 100
    """
    # Here we impose to duplicate the number
    num = np.maximum(np.max(ndecade * np.log10(energy_max / energy_min)), 2)
    energy = np.geomspace(energy_min, energy_max, num=int(num), axis=-1)
    integral = trapz_loglog(energy*func(energy), energy, axis=-1)
    return integral.sum(axis=0)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class GCsource:
    """A class to encapsulate Guo13 model of the GC source.
    
    For now only the TeV part is taken into account.
    
    Parameters:
    W0 : `astropy.units.Quantity`
        the energy injected. Default is 1e48 erg.
    alpha : `astropy.units.Quantity`
        the source proton index. Default is 1.8
    e_cutoff : `astropy.units.Quantity`
        the source cutoff proton energy. Default is 1 PeV
    M_target :`astropy.units.Quantity`
        the target cloud mass. Default is 1e5 M_sun.
    R_target : `astropy.units.Quantity`
        the target cloud radius. Default is 1.5 pc.
    d_target : `astropy.units.Quantity`
        the target cloud distance. Default is 8.2 kpc.        
    D0 :  `astropy.units.Quantity`
        the diffusion coefficient normalization. Default is 1e27 cm2/s.
    Eref : `astropy.units.Quantity`
        the diffusion coefficient reference energy. Default is 10 GeV.
    diff_index : `astropy.units.Quantity`
        the diffusion coefficient index. Default is 1/3.
    
    """
    def __init__(self, 
                 cloud_atlas,
                 base_geom,
                 binsize_pc,
                 W0="1e48 erg", 
                 alpha=1.8,
                 e_cutoff="1 PeV",
                 #M_target="1e5 M_sun" , #inutile
                 #R_target="30 pc", 
                 d_target="8.122 kpc",
                 D0="1e27 cm2/s", 
                 Eref="10 GeV",
                 diff_index=0.3):
        
        self.W0 = u.Quantity(W0)
        self.alpha = u.Quantity(alpha)
        self.e_cutoff = u.Quantity(e_cutoff)
        self.D0 = u.Quantity(D0)
        self.Eref = u.Quantity(Eref)
        self.diff_index = u.Quantity(diff_index)
        
        self.base_geom = base_geom
        self.binsize_pc = binsize_pc
        #self.M_target = u.Quantity(M_target)
        #self.R_target = u.Quantity(R_target)
        self.d_target = u.Quantity(d_target)
        
        self.cloud_atlas = cloud_atlas
        
        self.injection_model = ExponentialCutoffPowerLaw(
                        amplitude=1/u.TeV,
                        e_0 = 100*u.TeV, #?
                        alpha = self.alpha,
                        e_cutoff = self.e_cutoff
        )
        self.injection_model.amplitude *= self.compute_norm(self.injection_model,self.W0)

        
    #@property
    #def volume(self):
    #    """Return target cloud volume."""
    #    return 4./3.*np.pi*self.R_target**3
     
        
    #@property
    #def density(self):
    #    """Compute the average target density."""
    #    return self.M_target/cst.m_p/self.volume
     
        
    def diff_coeff(self, energy):
        """Compute diffusion coefficient at given  energy.
        
        Parameters
        ----------
        energy : `astropy.units.Quantity`
            energy.
        """
        coeff = self.D0 * (energy/self.Eref)**(self.diff_index.value)
        return coeff.to("cm2/s")
      
        
    def diff_radius(self, energy, time):
        """Compute diffusion radius at given time and energy.
        
        Parameters
        ----------
        time : `astropy.units.Quantity`
            time since injection. Must be a scalar quantity.
        energy : `astropy.units.Quantity`
            energy.
        """
        energy = np.atleast_1d(energy)
        return np.sqrt(2* self.diff_coeff(energy)*time).to("cm")
    
    
    def proton_spectrum(self,energy, time, distance):
        """Compute the proton spectrum at energy, distance and time from injection."""
        distance = np.atleast_1d(distance)
        distance_dims = len(distance.shape)
        axis = list(range(-distance_dims,0))
        
        distance = distance[np.newaxis, ...]
        rdiff = np.expand_dims(self.diff_radius(energy, time), axis=axis)
        
        injection = self.injection_model(energy)
        injection = np.expand_dims(injection, axis=axis)
        
        return injection*((np.sqrt(2*np.pi)*rdiff)**-3 * np.exp(-0.5*(distance/rdiff)**2))

                    
    def set_pion_spectrum_matrix2D_old(self, energy, time):#,
                                   #xbin, 
                                   #ybin, 
                                   #zbin):
        """Integrate proton spectrum over the line of sight."""
        self.proton_energy = energy
        
        xlin = np.linspace(-self.R_target, self.R_target, xbin)
        ylin = np.linspace(-self.R_target, self.R_target, ybin)
        zlin = np.linspace(-self.R_target, self.R_target, zbin)
        self.xlin = xlin
        self.ylin = ylin
        self.zlin = zlin
        
        self.nh_ref = 1/u.cm**3
        
        bin_volume = (2*self.R_target)**3/(xbin*ybin*zbin)
        xbin_length = 2*self.R_target/xbin
        
        X, Y, Z = np.meshgrid(xlin, ylin, zlin)
        
        self.local_density = calculate_local_density_allclouds(X, Y, Z, self.cloud_atlas)
        
        distance = np.sqrt(X**2 + Y**2 + Z**2)
        
        #log.info('- Computing "spectra"')
        spectra = self.proton_spectrum(energy, time, distance) 
        
        #log.info('- Computing "spectrum_3D"')
        spectrum_3D = spectra*bin_volume.to('cm3')*np.repeat(np.expand_dims(self.local_density, axis=0), len(energy), axis=0)/self.nh_ref
        
        
        #log.info('- Computing "integrated_spectrum')
        self.integrated_spectrum = np.sum(spectrum_3D, axis=2)

        gc.collect()
    
    
    def set_pion_spectrum_map(self, energy, time):
        self.proton_energy = energy
        
        center_pos = SkyCoord(self.base_geom.center_skydir, distance= self.base_geom.center_coord[2])
        center = center_pos.galactocentric
        
        coord = self.base_geom.get_coord()
        skyc = SkyCoord(coord['lon'], coord['lat'], frame=self.base_geom.frame, distance=coord['los']).galactocentric

        coord_los = skyc.x - center.x
        coord_lon = skyc.y - center.y
        coord_lat = skyc.z - center.z
        xbin, ybin, zbin = self.base_geom.data_shape
        self.x_space = np.linspace(-xbin*self.binsize_pc, xbin*self.binsize_pc, xbin)
        self.y_space = np.linspace(-ybin*self.binsize_pc, ybin*self.binsize_pc, ybin)
        self.z_space = np.linspace(-zbin*self.binsize_pc, zbin*self.binsize_pc, zbin)
        
        self.nh_ref = 1/u.cm**3
        bin_volume = self.binsize_pc**3
        
        X,Y,Z = coord_los.to('pc'), coord_lon.to('pc'), coord_lat.to('pc')
        
        self.local_density = calculate_local_density_allclouds(X, Y, Z, self.cloud_atlas)
        
        distance = np.sqrt(X**2 + Y**2 + Z**2)
        
        spectra = self.proton_spectrum(energy, time, distance) 
        
        spectrum_3D = spectra*bin_volume.to('cm3')*np.repeat(np.expand_dims(self.local_density, axis=0), len(energy), axis=0)/self.nh_ref
        #print(spectrum_3D.shape)
        self.integrated_spectrum = np.sum(spectrum_3D, axis=1)
        #print(self.integrated_spectrum.shape)
        gc.collect()
        
        
    def draw_density_maps(self):# éventuellement
        
        map_2D_density_np1 = np.sum(local_density, axis=1)
        map_2D_density_np0 = np.sum(local_density, axis=0)
        map_2D_density_np2 = np.sum(local_density, axis=2)

        map_2D_density_wcs_front = Map.create(binsz=binsize_deg, map_type='wcs', skydir=position, frame='icrs', npix=nbins_3d)
        
        map_2D_density_wcs_side = Map.create(binsz=binsize_deg, map_type='wcs', skydir=position, frame='icrs', npix=nbins_3d)# attention aux dimensions

        map_2D_density_wcs_front.data = map_2D_density_np0.value
        map_2D_density_wcs_front.write(pathres/f'density_{title}_yz.fits', overwrite=True)
        map_2D_density_wcs_side.data = map_2D_density_np1.value
        map_2D_density_wcs_side.write(pathres/f'density_{title}_zx.fits', overwrite=True)
        map_2D_density_wcs_side.data = map_2D_density_np2.value
        map_2D_density_wcs_side.write(pathres/f'density_{title}_xy.fits', overwrite=True)
    

    @staticmethod
    def compute_norm(model, W0, emin=1*u.GeV, emax=1*u.PeV, ndecade=100):
        """Compute model norm."""
        integral = integrate_energy_flux(model, emin, emax, ndecade)
        return (W0/integral).to('')
     
        
    def __call__(self, photon_energies, y, z):
        """Return flux at given energies."""
        idy, idz = find_nearest(self.y_space, y), find_nearest(self.z_space, z)
        
        model = TableModel(self.proton_energy, self.integrated_spectrum[:,idy,idz]) # can be 4D (energy on the 1st dim)

        pion_model = PionDecay(model, nh=self.nh_ref)
        
        return pion_model.flux(photon_energies, self.d_target)
    
    
    
def calculate_local_density_allclouds(X, Y, Z, cloud_atlas):
    #valid_map_tot = np.zeros_like(X, dtype=bool)/u.pc
    nh_map_tot = np.zeros_like(X)/(u.pc*u.cm**3)
    k = 0
    n = len(cloud_atlas)
    
    for cloud in cloud_atlas:
        k += 1
        log.info(f'- - Calculating density for cloud {k} of {n}')
        
        density = cloud.density.to('cm-3')

        valid_map = cloud.is_in_cloud(X, Y, Z)
        nh_map = valid_map*density
        
        nh_map_tot = nh_map_tot + nh_map
        
        del nh_map
        del valid_map
        gc.collect()

    return nh_map_tot


def calculate_local_density_1cloud(X,Y,Z, cloud):
    valid_map = np.zeros_like(X, dtype=bool)/u.pc
    nh_map = np.zeros_like(X)/(u.pc*u.cm**3)
    density = cloud.density.to('cm-3')

    valid_map = cloud.is_in_cloud(X, Y, Z)
    nh_map = valid_map*density
    
    return nh_map

        
def compute_map(cloud_atlas, 
                plane_bins, los_bins, binsize_pc,
                title, 
                time, 
                min_energy, max_energy, energy_bins,
                photon_min_energy, photon_max_energy, photon_energy_bins, 
                ):
    """
    """

    r_gal_cen = 8.122 * u.kpc
    binsize_deg = np.arctan(binsize_pc/r_gal_cen).to('deg')
    GC_pos = SkyCoord.from_name('SgrA*')
    
    xmax = binsize_pc*(los_bins+1)

    proton_energies = np.geomspace(min_energy, max_energy, energy_bins)*u.TeV # ~5 bins par decade, 
    
    los = MapAxis.from_edges(r_gal_cen+np.linspace(-xmax, xmax, los_bins+1), unit="pc", name="los")

    base_geom = WcsGeom.create(skydir=GC_pos, binsz=binsize_deg, npix=plane_bins, frame="galactic", axes=[los])
    
    source = GCsource(cloud_atlas, base_geom, binsize_pc)
    
    log.info('- Calculating pion spectrum matrix')
    source.set_pion_spectrum_map(proton_energies, time)
    
    photon_energies = np.geomspace(photon_min_energy, photon_max_energy, photon_energy_bins)*u.TeV
    map_np_fluxes = np.zeros((photon_energy_bins, plane_bins, plane_bins))
    
    photon_energy_axis = MapAxis.from_bounds(photon_min_energy, photon_max_energy, photon_energy_bins, name='energy', unit='TeV', interp='log')
    map_wcs_fluxes = Map.create(binsz=binsize_deg, map_type='wcs', skydir=GC_pos, frame='galactic',
                     npix=plane_bins, axes=[photon_energy_axis])
    
    #source.draw_density_maps()
    map_2D_density_np0 = np.sum(source.local_density, axis=0)
    map_2D_density_wcs = Map.create(binsz=binsize_deg, map_type='wcs', skydir=GC_pos, frame='galactic', npix=plane_bins)
    map_2D_density_wcs.data = map_2D_density_np0.value
    map_2D_density_wcs.write(pathres/f'density_{title}_yz.fits', overwrite=True)
    
    
    log.info('- Computing the VHE gamma map')
    tot = plane_bins**2
    k = 0
    for ny,y in enumerate(source.y_space):
        for nz,z in enumerate(source.z_space): #compter à l'envers ?
            spectrum_value = source(photon_energies, y.to_value('pc'),z.to_value('pc')).to('s-1 TeV-1 cm-2').value
            if np.isnan(spectrum_value[0]):
                map_np_fluxes[:,ny,nz] = np.zeros((photon_energy_bins))
            else:
                map_np_fluxes[:,ny,nz] = spectrum_value
            k+=1
            print(f'{k/tot*100:0.2f} %', end='\r')
            gc.collect()
    
    map_wcs_fluxes.data = map_np_fluxes
    
    map_wcs_fluxes.write(pathres/f'flux_{title}.fits', overwrite=True)



@cli.command("run_sim", help="Simulate a VHE gamma image of the GC")

@click.argument("n_atlas", default='1', type=int)
@click.argument("plane_bins", default='50', type=int)
@click.argument("los_bins", default='50', type=int)
@click.argument("min_time", default='100', type=float)
@click.argument("max_time", default='1000', type=float)
@click.argument("time_bins", default='1', type=int)
@click.argument("min_energy", default='1', type=float)
@click.argument("max_energy", default='1000', type=float)
@click.argument("energy_bins", default='20', type=int)
@click.argument("photon_min_energy", default='0.5', type=float)
@click.argument("photon_max_energy", default='50', type=float)
@click.argument("photon_energy_bins", default='15', type=int)
@click.argument("binsize_pc", default='0.25', type=float)

def run_sim(n_atlas, plane_bins, los_bins, min_time, max_time, time_bins, min_energy, max_energy, energy_bins,
            photon_min_energy, photon_max_energy, photon_energy_bins, binsize_pc):
    
    """Main command : Simulate  a VHE gamma image of the GC, or any cloud/group of clouds
    
    Parameters :
    n_atlas : int
        index of the cloud selection used for the simulation, see create_ferriere_clouds
    spacebins : int
        number of spatial bins in all 3 dimension
    min_time : float
        (in yrs) first (and potentially only) time at which the simulation is calculated
    max_time : float
        (in yrs) end of the array of times if several simulation at different times are performed
    time_bins : int
        number of time bins, should generally be 1
    min_energy : float
        (in TeV) beginning of the energy axis for the proton distribution
    max_energy : float
        (in TeV) end of the energy axis for the proton distribution
    energy_bins : int
        number of energy bins for the proton distribution
    """
    start = time.perf_counter()
    
    cloud_atlas = create_ferriere_clouds(n_atlas)
    
    times = np.geomspace(min_time*u.yr, max_time*u.yr, time_bins)
    for t in times:
        log.info(f'Simulating the GC at {t}')
        map_2D = compute_map(cloud_atlas, 
                    plane_bins=plane_bins, 
                    los_bins = los_bins,
                    title=f'real_clouds{n_atlas}_{plane_bins}bins_{int(np.floor(t.value))}yrs' ,
                    time=t,
                    min_energy=min_energy,
                    max_energy=max_energy,
                    energy_bins=energy_bins,
                    photon_min_energy=photon_min_energy,
                    photon_max_energy=photon_max_energy,
                    photon_energy_bins=photon_energy_bins,
                    binsize_pc=binsize_pc*u.pc)
        gc.collect()
        
    end = time.perf_counter()
    log.info(f'Time taken {end - start:0.1f}')

    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    cli()

    
