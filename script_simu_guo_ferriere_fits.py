
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
from gammapy.maps import Map,MapAxis
from gammapy.utils.integrate import trapz_loglog

from clouds import Cloud_sphere, Cloud_ring, Cloud_ellipsoid, Cloud_cylinder


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
                 W0="1e48 erg", 
                 alpha=1.8,
                 e_cutoff="1 PeV",
                 M_target="1e5 M_sun" , #inutile
                 R_target="30 pc", 
                 d_target="8.2 kpc",
                 D0="1e27 cm2/s", 
                 Eref="10 GeV",
                 diff_index=0.3):
        
        self.W0 = u.Quantity(W0)
        self.alpha = u.Quantity(alpha)
        self.e_cutoff = u.Quantity(e_cutoff)
        self.D0 = u.Quantity(D0)
        self.Eref = u.Quantity(Eref)
        self.diff_index = u.Quantity(diff_index)
        self.M_target = u.Quantity(M_target)
        self.R_target = u.Quantity(R_target)
        self.d_target = u.Quantity(d_target)
        self.cloud_atlas = cloud_atlas
        
        self.injection_model = ExponentialCutoffPowerLaw(
                        amplitude=1/u.TeV,
                        e_0 = 100*u.TeV, #?
                        alpha = self.alpha,
                        e_cutoff = self.e_cutoff
        )
        self.injection_model.amplitude *= self.compute_norm(self.injection_model,self.W0)

        
    @property
    def volume(self):
        """Return target cloud volume."""
        return 4./3.*np.pi*self.R_target**3
     
        
    @property
    def density(self):
        """Compute the average target density."""
        return self.M_target/cst.m_p/self.volume
     
        
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

                    
    def set_pion_spectrum_matrix2D(self, energy, time, 
                                   xbin, 
                                   ybin, 
                                   zbin):
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
    

    @staticmethod
    def compute_norm(model, W0, emin=1*u.GeV, emax=1*u.PeV, ndecade=100):
        """Compute model norm."""
        integral = integrate_energy_flux(model, emin, emax, ndecade)
        return (W0/integral).to('')
     
        
    def __call__(self, photon_energies, y, z):
        """Return flux at given energies."""
        #idy, idz = find_nearest(self.fov_width, y), find_nearest(self.fov_width, z)
        idy, idz = find_nearest(self.ylin, y), find_nearest(self.zlin, z)
        
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


def create_ferriere_clouds(n_atlas):
    SgrA_pos = (0,0,0) #origine
    SgrA_est_pos = (-2.0,1.2,-1.5)
    SC_pos = (8,-11,-5) #pas sûr pour x = 4-12 #souci sur l'axe z/dec
    EC_pos = (-3,7,-4.5)

    MR_pos = (3,-4,5) #b/w EC et SC
    SS_pos = (3,-4,0) #b/w SC et CNR
    WS_pos = (-2,-2,-2) #W bdy of SNR
    NR_pos = (-3,5,2) #N bdy of SNR
    
    
    CNR_rot = np.pi/2*np.array([+1/2.5,1/6,0]) #à peu près
    SNR_rot = [0,0,0] #pas important
    SC_rot = np.pi/2*np.array([1/6,0,0]) #ok

    MR_rot = np.pi/2*np.array([1/4,1/4,-1/4])
    SS_rot = np.pi/2*np.array([0,0,-1/4])
    WS_rot =np.pi/2*np.array([0,0,0])
    NR_rot = np.pi/2*np.array([0,0,0]) # pas important
    
    
    mCC = (190+12+160)*u.M_sun
    mCNR = 2e5*u.M_sun
    mSNR = 19*u.M_sun
    mhalo = 13000*u.M_sun
    mSC = 2.2e5*u.M_sun
    mEC = 1.9e5*u.M_sun

    mMR = 6e4*u.M_sun
    mSS = 1.6e4*u.M_sun
    mWS = 4.5e3*u.M_sun
    mNR = 2.2e3*u.M_sun
    
    CC = Cloud_ellipsoid(SgrA_pos, mCC, 
                     '2.9 pc','2.9 pc','2.1 pc')
    CNR = Cloud_ring(SgrA_pos, mCNR, rot_vec=CNR_rot)
    SNR = Cloud_ellipsoid(SgrA_est_pos, mSNR,
                         '9.0 pc', '9.0 pc', '6.7 pc',
                         rot_vec=SNR_rot)
    halo = Cloud_sphere(SgrA_est_pos, mhalo, '9 pc')
    SC = Cloud_ellipsoid(SC_pos, mSC,
                        '7.5 pc', '15 pc', '7.5 pc',
                        rot_vec=SC_rot)
    EC = Cloud_sphere(EC_pos, mEC, '4.5 pc')
    MR = Cloud_cylinder(MR_pos, mMR, '9 pc', '1 pc',
                   rot_vec=MR_rot)
    SS = Cloud_cylinder(SS_pos, mSS, '7 pc', '1 pc',
                       rot_vec=SS_rot)
    WS = Cloud_cylinder(WS_pos, mWS, '8 pc', '0.5 pc',
                       rot_vec=WS_rot)
    NR = Cloud_cylinder(NR_pos, mNR, '4 pc', '0.5 pc',
                       rot_vec=NR_rot)
    
    test = Cloud_sphere((0,5,5), mCNR, '5 pc')
    test2 = Cloud_ellipsoid((0,-5,-5), mCNR, '2.9 pc','2.9 pc','2.1 pc', np.pi/2*np.array([1/6,0,0]))
    
    if n_atlas==0 :
        return [test, test2]
    if n_atlas==1 :
        return [CC, CNR, SNR, halo, SC, EC]
    if n_atlas==2 :
        return [CC, CNR, SNR, halo, SC, EC, MR, SS, WS, NR] #plus complet, pas forcément utile
 
        
def compute_map(cloud_atlas, 
                nbins_3d, 
                title, 
                time, 
                min_energy,max_energy,energy_bins,
                photon_min_energy,photon_max_energy, photon_energy_bins, 
                R_target):
    """
    """
    photon_energy_bins = 10 #en attendant
    R_target = 30*u.pc
    

    energies = np.geomspace(min_energy, max_energy, energy_bins)*u.TeV # ~5 bins par decade, 
    source = GCsource(cloud_atlas,R_target=R_target)
    
    log.info('- Calculating pion spectrum matrix')
    source.set_pion_spectrum_matrix2D(energies, time, nbins_3d, nbins_3d, nbins_3d)
        
    map_3D = np.zeros((photon_energy_bins, len(source.ylin), len(source.zlin)))
    hess_energies = np.geomspace(photon_min_energy, photon_max_energy, photon_energy_bins)*u.TeV
    
    
    #position = SkyCoord(17.45, -29.0, frame='icrs', unit='deg')# appelé Sgr A* directement
    position = SkyCoord.from_name('SgrA*')
    energy_axis = MapAxis.from_bounds(photon_min_energy, photon_max_energy, photon_energy_bins, name='energy', unit='TeV')
    
    width = 2*R_target/source.d_target*u.rad
    binsize = width/nbins_3d
    
    map_wcs = Map.create(binsz=binsize.to('deg'), map_type='wcs', skydir=position, frame='icrs',
                     width=width.to('deg'), axes=[energy_axis])
    

    map_2D_density_np = np.sum(source.local_density, axis=1)

    map_2D_density_wcs = Map.create(binsz=binsize.to('deg'), map_type='wcs', skydir=position, frame='icrs', width=width.to('deg'))
    map_2D_density_wcs.data = map_2D_density_np.value
    map_2D_density_wcs.write(pathres/f'density_{title}.fits', overwrite=True)
    
    
    log.info('- Computing the VHE gamma map')
    tot = len(source.ylin)*len(source.zlin)
    k = 0
    for ny,y in enumerate(source.ylin):
        for nz,z in enumerate(source.zlin):
            spectrum_value = source(hess_energies, y.to_value('pc'),z.to_value('pc')).to('s-1 TeV-1 cm-2').value
            if np.isnan(spectrum_value[0]):
                map_3D[:,ny,nz] = np.zeros((photon_energy_bins))
            else:
                map_3D[:,ny,nz] = spectrum_value
            k+=1
            print(f'{k/tot*100:0.2f} %', end='\r')
            gc.collect()
    
    map_wcs.data = map_3D #problème ici
    
    map_wcs.write(pathres/f'flux_{title}.fits', overwrite=True)



@cli.command("run_sim", help="Simulate a VHE gamma image of the GC")
@click.argument("n_atlas", default='1', type=int)
@click.argument("spacebins", default='20', type=int)
@click.argument("min_time", default='100', type=float)
@click.argument("max_time", default='1000', type=float)
@click.argument("time_bins", default='1', type=int)
@click.argument("min_energy", default='1', type=float)
@click.argument("max_energy", default='1000', type=float)
@click.argument("energy_bins", default='20', type=int)
@click.argument("photon_min_energy", default='0.5', type=float)
@click.argument("photon_max_energy", default='50', type=float)
@click.argument("photon_energy_bins", default='15', type=int)
@click.argument("R_target", default='30', type=int)

def run_sim(n_atlas, spacebins, min_time, max_time, time_bins, min_energy, max_energy, energy_bins,
            photon_min_energy, photon_max_energy, photon_energy_bins, r_target):
    
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
                    nbins_3d=spacebins, 
                    title=f'real_clouds{n_atlas}_{spacebins}bins_{int(np.floor(t.value))}yrs' ,
                    time=t,
                    min_energy=min_energy,
                    max_energy=max_energy,
                    energy_bins=energy_bins,
                    photon_min_energy=photon_min_energy,
                    photon_max_energy=photon_max_energy,
                    photon_energy_bins=photon_energy_bins,
                    R_target=r_target*u.pc)
        gc.collect()
        
    end = time.perf_counter()
    log.info(f'Time taken {end - start:0.1f}')

    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    cli()

    
