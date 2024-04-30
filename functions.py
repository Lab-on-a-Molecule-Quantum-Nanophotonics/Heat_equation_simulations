import numpy as np
import matplotlib.pyplot as plt

def square_lattice( Nx, Ny ):
    N = Nx*Ny
    R = np.zeros(shape=(N,2))
    n=0
    for y in range(Ny):
        for x in range(Nx):
            R[n][0] = x
            R[n][1] = y
            n += 1
    return R

def triangular_lattice( Nx, Ny ):
    N = Nx*Ny
    R = np.zeros(shape=(N,2))
    n=0
    for y in range(Ny):
        for x in range(Nx):
            if y%2==0:
                R[n][0] = x
                R[n][1] = y*(np.sqrt(3.)/2.)
            else:
                R[n][0] = x + .5
                R[n][1] = y*(np.sqrt(3.)/2.)
            n += 1

    return R

def honeycomb_lattice( Nx, Ny ):
    N = Nx*Ny
    R = np.zeros(shape=(2*N,2))
    n=0
    for y in range(Ny):
        for x in range(Nx):
            if y%2==0:
                #R[n][0] = x
                #R[n][1] = y*(np.sqrt(3.)/2.)
                #R[n+1][0] = x
                #R[n+1][1] = y*(np.sqrt(3.)/2.) + 1./np.sqrt(3.)
                R[n][0] = x/(np.sqrt(3.)/2.)*1.5
                R[n][1] = y*1.5
                R[n+1][0] = x/(np.sqrt(3.)/2.)*1.5
                R[n+1][1] = y*1.5 + 1.
            else:
                #R[n][0] = x + .5
                #R[n][1] = y*(np.sqrt(3.)/2.)
                #R[n+1][0] = x + .5
                #R[n+1][1] = y*(np.sqrt(3.)/2.) + 1./np.sqrt(3.)
                R[n][0] = (x + .5)/(np.sqrt(3.)/2.)*1.5
                R[n][1] = y*1.5
                R[n+1][0] = (x + .5)/(np.sqrt(3.)/2.)*1.5
                R[n+1][1] = y*1.5 + 1.
            n += 2

    return R


def hexagonal_boundaries(R):

    Xmax = np.amax(R[:,0])
    Ymax = np.amax(R[:,1])+1

    newR = []
    for n in range(int(R.shape[0])):
        if R[n][1] <= Ymax/2. :
            if (R[n][1] + 2*(Ymax/Xmax)*R[n][0])>Ymax/2. and (R[n][1] - 2*(Ymax/Xmax)*R[n][0])> -3.*Ymax/2.:
                newR.append( R[n] )
        elif R[n][1] > Ymax/2. :
            if (R[n][1] + 2*(Ymax/Xmax)*R[n][0])<5.*Ymax/2. and (R[n][1] - 2*(Ymax/Xmax)*R[n][0])<Ymax/2.:
                newR.append( R[n] )

    return np.asarray(newR)

def plot_lattice(R):
    plt.figure(figsize=(8,8))
    plt.plot( R[:,0], R[:,1], 'bo'  )
    plt.grid()
    plt.axis([0, np.amax(R[:,0])+.5, 0, np.amax(R[:,1])+.5])
    #plt.axis([10, 16, 10, 16])
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 26,
            }
    plt.xlabel('x', fontdict=font)
    plt.ylabel('y', fontdict=font)
    plt.show()
    
def plot_molecule_position_and_laser_on_lattice(R, X_meas, Y_meas, Y_L):
    n_meas = np.argmin( np.square(R[:,0]-X_meas) + np.square(R[:,1]-Y_meas) )
    n_L = np.argmin( np.square(R[:,0]-X_meas) + np.square(R[:,1]-Y_L) )
    plt.figure(figsize=(8,8))
    plt.scatter( R[:,0], R[:,1]  )
    plt.scatter( R[n_meas,0], R[n_meas,1]  )
    plt.scatter( R[n_L,0], R[n_L,1]  )
    plt.grid()
    plt.axis([0, np.amax(R[:,0])+.5, 0, np.amax(R[:,1])+.5])
    #plt.axis([10, 16, 10, 16])
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 26,
            }
    plt.xlabel('x', fontdict=font)
    plt.ylabel('y', fontdict=font)
    plt.show()
    
def heat_kernel(R, vec):
    M = R.shape[0]
    J = np.zeros(shape=(M,M))

    for m1 in range(M):
        J[m1][m1] = - 4. - vec[m1]       
        for m2 in range(M):
            dR = np.sqrt( np.square(R[m1][0]-R[m2][0]) + np.square(R[m1][1]-R[m2][1]) )
            if dR < 1.1 and dR > 0.1:
                J[m1][m2] = 1./dR**2.
    #evals, evecs = np.linalg.eig(J)
    #idx = evals.argsort()
    #evals = evals[idx]
    #evecs = evecs[:,idx]
    #Emax = np.amax(evals)
    #Emin = np.amin(evals)

    #BW = Emax - Emin

    #evals = (evals-Emin)/BW
    #J = (J - Emin)/BW
    invJ = np.linalg.inv(J)
    
    return invJ 
                
def x_laser_scan(R, invJ, X_meas, Y_meas, Yc, C, nu, sigma_Q):
    Xmax = np.amax(R[:,0])
    Xmin = np.amin(R[:,0])
    Ymax = np.amax(R[:,1])

    n_meas = np.argmin( np.square(R[:,0]-X_meas) + np.square(R[:,1]-Y_meas) )

    iter_X = 101
    dX = (Xmax-Xmin)/iter_X
    T_meas = []
    arr_X = []
    for nx in range(iter_X):
        Xc = dX*nx + Xmin

        cQ = np.zeros(shape=(R.shape[0]))
        cQ = np.exp( - ( np.square(R[:,0]-Xc) + np.square(R[:,1]-Yc) )/(2.*sigma_Q**2.) )/(2.*np.pi*sigma_Q**2.)

        T_field = -np.matmul( invJ, cQ )
        T_meas.append( T_field[n_meas] )
        arr_X.append(Xc)
        
    arr_X = np.asarray(arr_X)
    T_meas = np.asarray(T_meas)
    T_meas = np.power( 1. + C*(T_meas-0*np.amin(T_meas)), 1./(1.+nu) ) 
    
    return arr_X, T_meas

def plot_dimensional_temperature(dr, arr_X, T_meas, T0, Tmax):
    
    
    arr_X = arr_X*dr
    T_meas = T_meas*T0

    plt.figure(figsize=(8,8))
    plt.plot( arr_X,  T_meas )
    plt.grid()
    plt.axis([np.amin(arr_X), np.amax(arr_X), 0, Tmax])
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 26,
            }
    plt.xlabel('micron', fontdict=font)
    plt.ylabel('Kelvin', fontdict=font)
    plt.show()

    

