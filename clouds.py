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
    
    def is_in_cloud(self, X, Y, Z):
        x0, y0, z0 = self.center

        xwidth = X[0,1,0] - X[0,0,0]
        ywidth = Y[1,0,0] - Y[0,0,0]
        zwidth = Z[0,0,1] - Z[0,0,0]

        Xp, Yp, Zp = X/u.pc - x0 + xwidth.value/2, Y/u.pc - y0 + ywidth.value/2, Z/u.pc - z0 + zwidth.value/2
        #Xp, Yp, Zp = X - x0 + xwidth/2, Y - y0 + ywidth/2, Z - z0 + zwidth/2

        return np.sqrt(Xp**2 + Yp**2 + Zp**2) < self.radius.value
    
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
    
    def is_in_cloud(self, X, Y, Z):
        x0, y0, z0 = self.center

        xwidth = X[0,1,0] - X[0,0,0]
        ywidth = Y[1,0,0] - Y[0,0,0]
        zwidth = Z[0,0,1] - Z[0,0,0]

        Xp, Yp, Zp = X/u.pc - x0 + xwidth.value/2, Y/u.pc - y0 + ywidth.value/2, Z/u.pc - z0 + zwidth.value/2
        #Xp, Yp, Zp = X - x0 + xwidth/2, Y - y0 + ywidth/2, Z - z0 + zwidth/2
        
        mat = np.array([Xp.ravel(), Yp.ravel(), Zp.ravel()]).T

        mat_rot = self.rot_matrix.apply(mat)

        Xp = mat_rot[...,0].reshape(X.shape)
        Yp = mat_rot[...,1].reshape(Y.shape)
        Zp = mat_rot[...,2].reshape(Z.shape)
        
        return np.sqrt( (Xp/self.lx.value)**2 + (Yp/self.ly.value)**2 + (Zp/self.lz.value)**2) < 1
        
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
    
    def is_in_cloud(self, X, Y, Z):
        x0, y0, z0 = self.center

        xwidth = X[0,1,0] - X[0,0,0]
        ywidth = Y[1,0,0] - Y[0,0,0]
        zwidth = Z[0,0,1] - Z[0,0,0]

        Xp, Yp, Zp = X/u.pc - x0 + xwidth.value/2, Y/u.pc - y0 + ywidth.value/2, Z/u.pc - z0 + zwidth.value/2
        #Xp, Yp, Zp = X - x0 + xwidth/2, Y - y0 + ywidth/2, Z - z0 + zwidth/2

        mat = np.array([Xp.ravel(), Yp.ravel(), Zp.ravel()]).T

        mat_rot = self.rot_matrix.apply(mat)

        Xp = mat_rot[...,0].reshape(X.shape)
        Yp = mat_rot[...,1].reshape(Y.shape)
        Zp = mat_rot[...,2].reshape(Z.shape)
        
        r = np.sqrt(Xp**2 + Yp**2)#*u.pc
        h = self.hin.value + (self.hout.value - self.hin.value)*(r - self.rin.value)/(self.rout.value - self.rin.value)

        p1 = r < self.rout.value
        p2 = r > self.rin.value
        p3 = Zp < h/2
        p4 = Zp > -h/2
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
    
    def is_in_cloud(self, X, Y, Z):
        x0, y0, z0 = self.center

        xwidth = X[0,1,0] - X[0,0,0]
        ywidth = Y[1,0,0] - Y[0,0,0]
        zwidth = Z[0,0,1] - Z[0,0,0]

        Xp, Yp, Zp = X/u.pc - x0 + xwidth.value/2, Y/u.pc - y0 + ywidth.value/2, Z/u.pc - z0 + zwidth.value/2
        #Xp, Yp, Zp = X - x0 + xwidth/2, Y - y0 + ywidth/2, Z - z0 + zwidth/2

        mat = np.array([Xp.ravel(), Yp.ravel(), Zp.ravel()]).T

        mat_rot = self.rot_matrix.apply(mat)

        Xp = mat_rot[...,0].reshape(X.shape)
        Yp = mat_rot[...,1].reshape(Y.shape)
        Zp = mat_rot[...,2].reshape(Z.shape)
        
        
        p1 = Zp < self.l.value/2
        p2 = Zp > -self.l.value/2
        p3 = np.sqrt(Xp**2 + Yp*2) < self.r.value
        
        return p1*p2*p3
    
    @lazyproperty
    def coords(self):
        pos = np.linspace(-self.max_radius, self.max_radius, self.nbins)
        coord = Galactocentric(x=pos[:, np.newaxis, np.newaxis], 
                           y=pos[np.newaxis,:,np.newaxis], 
                           z=pos[np.newaxis, np.newaxis, :])

def create_test_clouds(n_atlas):
    test = Cloud_sphere((0,5,5), 2*mCNR, '5 pc')
    test2 = Cloud_ellipsoid((0,-5,-5), mCNR, '3 pc','3 pc','2 pc', np.pi/2*np.array([1/6,0,0]))
    
    if n_atlas==0 :
        return [test, test2]
    
    
def create_ferriere_clouds(n_atlas):
    SgrA_pos = (0,0,0) #origine
    SgrA_est_pos = (-2.0,1.2,-1.5)
    SC_pos = (8,-11,-5) #pas sûr pour x = 4-12 #souci sur l'axe z/dec
    EC_pos = (-3,7,-4.5)

    MR_pos = (3,-4,5) #b/w EC et SC
    SS_pos = (3,-4,0) #b/w SC et CNR
    WS_pos = (-2,-2,-2) #W bdy of SNR
    NR_pos = (-3,5,2) #N bdy of SNR
    
    
    #CNR_rot = np.pi/2*np.array([+1/2.5,1/6,0]) #à peu près
    CNR_rot = [0,0,0]
    SNR_rot = [0,0,0] #pas important
    #SC_rot = np.pi/2*np.array([1/6,0,0]) #ok
    SC_rot = [0,0,0]

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
                         '9.0 pc', '9.0 pc', '6.7 pc',rot_vec=SNR_rot)
    halo = Cloud_sphere(SgrA_est_pos, mhalo, '9 pc')
    
    SC = Cloud_ellipsoid(SC_pos, mSC,
                        '7.5 pc', '15 pc', '7.5 pc',rot_vec=SC_rot)
    EC = Cloud_sphere(EC_pos, mEC, '4.5 pc')
    
    
    MR = Cloud_cylinder(MR_pos, mMR, '9 pc', '1 pc',
                   rot_vec=MR_rot)
    SS = Cloud_cylinder(SS_pos, mSS, '7 pc', '1 pc',
                       rot_vec=SS_rot)
    WS = Cloud_cylinder(WS_pos, mWS, '8 pc', '0.5 pc',
                       rot_vec=WS_rot)
    NR = Cloud_cylinder(NR_pos, mNR, '4 pc', '0.5 pc',
                       rot_vec=NR_rot)
    
    test = Cloud_sphere((2,5,7), mCNR, '4 pc')
    test2 = Cloud_ellipsoid((-2,-5,-7), mCNR, '3 pc','3 pc','2 pc', np.pi/2*np.array([1/6,0,0]))
    
    if n_atlas==0 :
        return [test, test2]
    if n_atlas==1 :
        return [CC, CNR, SNR, halo, SC, EC]
    if n_atlas==2 :
        return [CC, CNR, SNR, halo, SC, EC, MR, SS, WS, NR] #plus complet, pas forcément utile
    if n_atlas==3 :
        return [EC, SC] # pour tester