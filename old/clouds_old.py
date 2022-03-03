import numpy as np
import astropy.units as u
import astropy.constants as cst
from astropy.utils import lazyproperty
from astropy.coordinates import Galactocentric
from astropy.table import Table

from scipy.spatial.transform import Rotation as R


class Cloud_sphere:
    """Class to represent a gas cloud.
    
    Parameters
    ----------
    center : `~astropy.coordinates.Galactocentric`
        the cloud center position.
    mass :`astropy.units.Quantity`
        the target cloud mass. Default is 1e5 M_sun.
    radius : `astropy.units.Quantity`
        the target cloud radius. Default is 1.5 pc.
    nbins : int
        the number of bins along one axis to describe the cloud. Default 50.
    """
    def __init__(self, 
                 center,
                 mass="1e5 M_sun" , 
                 radius="2 pc",
                 nbins=50,
                ):
        self.center = center
        
        self.mass = u.Quantity(mass)
        self.radius = u.Quantity(radius)
        
        self.nbins = nbins
        self.rot_matrix = None

    @property
    def volume(self):
        """Return target cloud volume."""
        return 4./3.*np.pi*self.radius**3
        
    @property
    def density(self):
        """Compute the average target density."""
        return self.mass/cst.m_p/self.volume 
    
    def is_in_cloud(self,x,y,z):
        return np.sqrt(x**2+y**2+z**2) < self.radius.value
    
    @lazyproperty
    def coords(self):
        pos = np.linspace(-self.radius, self.radius, self.nbins)
        coord = Galactocentric(x=pos[:, np.newaxis, np.newaxis], 
                           y=pos[np.newaxis,:,np.newaxis], 
                           z=pos[np.newaxis, np.newaxis, :])
        return coord

    
class Cloud_ellipsoid:
    """Class to represent a gas cloud.
    
    Parameters
    ----------
    center : `~astropy.coordinates.Galactocentric`
        the cloud center position.
    mass :`astropy.units.Quantity`
        the target cloud mass. Default is 1e5 M_sun.
    lx,ly,lz : `astropy.units.Quantity`
        the target cloud lengths, default is 2 pc
    nbins : int
        the number of bins along one axis to describe the cloud. Default 50.
    """
    def __init__(self, 
                 center,
                 mass="1e5 M_sun" , 
                 lx="2 pc",
                 ly='2 pc',
                 lz='2 pc',
                 rot_vec = [0,0,0],
                 nbins=50,
                ):
        self.center = center
        
        self.mass = u.Quantity(mass)
        self.lx = u.Quantity(lx)
        self.ly = u.Quantity(ly)
        self.lz = u.Quantity(lz)
        
        self.rot_matrix = R.from_rotvec(np.array(rot_vec))
        self.rot_vecx = rot_vec[0] 
        self.rot_vecy = rot_vec[1]
        self.rot_vecz = rot_vec[2]
        
        self.max_radius = 1*u.pc*np.max([self.lx.value,self.ly.value,self.lz.value])/2
        
        self.nbins = nbins

    @property
    def volume(self):
        """Return target cloud volume."""
        return 1./6.*np.pi*self.lx*self.ly*self.lz
    
    def is_in_cloud(self,x,y,z):
        return np.sqrt( (x/self.lx.value)**2 + (y/self.ly.value)**2 + (z/self.lz.value)**2) < 1
        
    @property
    def density(self):
        """Compute the average target density."""
        return self.mass/cst.m_p/self.volume 
    
    @lazyproperty
    def coords(self):
        pos = np.linspace(-self.max_radius, self.max_radius, self.nbins)
        coord = Galactocentric(x=pos[:, np.newaxis, np.newaxis], 
                           y=pos[np.newaxis,:,np.newaxis], 
                           z=pos[np.newaxis, np.newaxis, :])
        

class Cloud_ring:
    """Class to represent a gas cloud.
    
    Parameters
    ----------
    center : `~astropy.coordinates.Galactocentric`
        the cloud center position.
    mass :`astropy.units.Quantity`
        the target cloud mass. Default is 1e5 M_sun.
    lx,ly,lz : `astropy.units.Quantity`
        the target cloud lengths, default is 2 pc
    nbins : int
        the number of bins along one axis to describe the cloud. Default 50.
    """
    def __init__(self, 
                 center,
                 mass="2e5 M_sun" , 
                 rin="1.2 pc",
                 rout='3 pc',
                 hin='0.4 pc',
                 hout='1.0 pc',
                 rot_vec = [0,0,0],
                 nbins=50,
                ):
        self.center = center
        
        self.mass = u.Quantity(mass)
        self.rin = u.Quantity(rin)
        self.rout = u.Quantity(rout)
        self.hin = u.Quantity(hin)
        self.hout = u.Quantity(hout)
        
        self.rot_matrix = R.from_rotvec(np.array(rot_vec))
        self.rot_vecx = rot_vec[0] 
        self.rot_vecy = rot_vec[1]
        self.rot_vecz = rot_vec[2]
        
        self.max_radius = u.Quantity(rout)
        
        self.nbins = nbins

    @property
    def volume(self):
        """Return target cloud volume."""
        #return (rout+rin)*np.pi*((hin+hout)*(rout-rin))/2
        return 18*u.pc**3 #default
    
    def is_in_cloud(self,x,y,z):
        r = np.sqrt(x**2 + y**2)#*u.pc
        z = z#*u.pc
        h = self.hin.value + (self.hout.value - self.hin.value)*(r - self.rin.value)/(self.rout.value - self.rin.value)

        p1 = r.value < self.rout.value
        p2 = r.value > self.rin.value
        p3 = z.value < h/2
        p4 = z.value > -h/2
        return p1*p2*p3*p4
    
    @property
    def density(self):
        """Compute the average target density."""
        return self.mass/cst.m_p/self.volume 
    
    @lazyproperty
    def coords(self):
        pos = np.linspace(-self.max_radius, self.max_radius, self.nbins)
        coord = Galactocentric(x=pos[:, np.newaxis, np.newaxis], 
                           y=pos[np.newaxis,:,np.newaxis], 
                           z=pos[np.newaxis, np.newaxis, :])
        
        
class Cloud_cylinder:
    """Class to represent a gas cloud.
    
    Parameters
    ----------
    center : `~astropy.coordinates.Galactocentric`
        the cloud center position.
    mass :`astropy.units.Quantity`
        the target cloud mass. Default is 1e5 M_sun.
    lx,ly,lz : `astropy.units.Quantity`
        the target cloud lengths, default is 2 pc
    nbins : int
        the number of bins along one axis to describe the cloud. Default 50.
    """
    def __init__(self, 
                 center,
                 mass="1e5 M_sun" , 
                 l="5 pc",
                 r='1 pc',
                 rot_vec = [0,0,0],
                 nbins=50,
                ):
        self.center = center
        
        self.mass = u.Quantity(mass)
        self.l = u.Quantity(l)
        self.r = u.Quantity(r)
        
        self.rot_matrix = R.from_rotvec(np.array(rot_vec))
        self.rot_vecx = rot_vec[0] 
        self.rot_vecy = rot_vec[1]
        self.rot_vecz = rot_vec[2]
        
        self.max_radius = np.max([self.l.value/2,self.r.value])
        
        self.nbins = nbins

    @property
    def volume(self):
        """Return target cloud volume."""
        return np.pi*self.r**2*self.l
        
    @property
    def density(self):
        """Compute the average target density."""
        return self.mass/cst.m_p/self.volume
    
    def is_in_cloud(self,x,y,z):
        #x,y,z = x*u.pc, y*u.pc, z*u.pc
        p1 = z.value < self.l.value/2
        p2 = z.value > -self.l.value/2
        p3 = np.sqrt(x.value**2 + y.value**2) < self.r.value
        return p1*p2*p3
    
    @lazyproperty
    def coords(self):
        pos = np.linspace(-self.max_radius, self.max_radius, self.nbins)
        coord = Galactocentric(x=pos[:, np.newaxis, np.newaxis], 
                           y=pos[np.newaxis,:,np.newaxis], 
                           z=pos[np.newaxis, np.newaxis, :])