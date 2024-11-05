import numpy as np
import mpmath
import time
import matplotlib
import matplotlib.pyplot as plt
from mpmath import coulombf, coulombg, log, conj
from scipy.special import gamma, gammainc, erf, eval_legendre, roots_legendre
from scipy.integrate import quad
from numerov import *


#==========================================
#Physical parameters

hbarc = 197.3269804
m = 938.918
mu = m/2
hh = hbarc**2 / (2*mu)
e2 = 0#1.44 
z1 = 1
z2 = 1

channel_radius = 5 #fm
l = 0 #orbital momentum 

#==========================================
#Potential 

def HO(r,b):
    #Harmonic oscilator 
    omega = 1./(mu*b**2)*hbarc**2
    return 1/2*mu*omega**2*r**2/hbarc**2


def Minnesota_singlet(r):
    return 200*np.exp(-1.487*r**2)-91.85*np.exp(-0.465*r**2)


def Coulomb(r):
    return z1*z2*e2/r


def EFT_pp(r, L, C, E0, E1):
    return C * np.exp(-L**2/4 * r**2) + 2*E0 * np.exp(-L**2/4 * r**2) + E1 * np.exp(-L**2/4 * r**2)

#==========================================
#Coulomb functions

def eta(energy):
    k=np.sqrt(2*mu*energy)
    eta = e2*mu/(k*hbarc)

    return eta


def coulombF(r,energy,l):
    #regular coulomb function
    k=np.sqrt(2*mu*energy)
    eta = e2*mu/(k*hbarc)

    return coulombf(l,eta,r)


def coulombG(r,energy,l):
    #irregular coulomb function
    k=np.sqrt(2*mu*energy)
    eta = e2*mu/(k*hbarc)

    return coulombg(l,eta,r)


def coulombH_minus(r, energy, l, d = False):
    
    dr = 10**(-10)
    
    if d == False:
        H = coulombG(r, energy, l) - 1j*coulombF(r, energy, l)
        return H 
    
    if d == True:
        H_d_plus = coulombG(r+dr/2, energy, l) - 1j*coulombF(r+dr/2, energy, l)
        H_d_minus = coulombG(r-dr/2, energy, l) - 1j*coulombF(r-dr/2, energy, l)
        return (H_d_plus - H_d_minus) / dr


def coulombH_plus(r, energy, l, d = False):
    
    dr = 10**(-10)
    
    if d == False:
        H = coulombG(r, energy, l) + 1j*coulombF(r, energy, l)
        return H 
    
    if d == True:
        H_d_plus = coulombG(r+dr/2, energy, l) + 1j*coulombF(r+dr/2, energy, l)
        H_d_minus = coulombG(r-dr/2, energy, l) + 1j*coulombF(r-dr/2, energy, l)
        return (H_d_plus - H_d_minus) / dr

#==========================================
#Lagange mesh 

def roots(N):  
    
    zeros = roots_legendre(N)[0]
    return (zeros + 1) / 2
        

def lagrange(N): 
    #Lagrange functions evaluated at channel radius 
    a = channel_radius
    zeros = roots(N)
    phi = np.empty(len(zeros))

    for i in range(len(zeros)):
        
        x = zeros[i]
        phi[i] = (-1)**(N+i+1) * (1/x) * (a*x*(1-x))**(1/2)*1/(a-a*x)
        
    return phi


def C_matrix_lagrange(zeros, a, N, E, potential, l=0): 
	
    C = np.empty((N, N))

    for i in range(len(zeros)):
        for j in range(len(zeros)):
            xi = zeros[i]
            xj = zeros[j]
            if xi!=xj:
                C[i,j] = hh * ( 
                        (-1)**(i+j) / (a**2*(xi*xj*(1-xi)*(1-xj))**(0.5))*
                        (N**2+N+1+(xi+xj-2*xi*xj)/(xi-xj)**2 - 1/(1-xi) - 1/(1-xj)) )
                
            elif xi==xj:
                C[i,j] = ( hh*((4*N**2+4*N+3)*xi*(1-xi)-6*xi+1) / (3*a**2*xi**2*(1-xi)**2)
                         + potential(a*xi) + hh * l*(l+1)/(a*xi)**2 + Coulomb(a*xi)- E )
			
    return C


def R_matrix_lagrange(energy, N, potential, l=0):

    zeros = roots(N)
	
    basis = lagrange(N)

    dim = len(zeros)

    R = np.empty((dim,dim))

    C = C_matrix_lagrange(zeros, channel_radius, N, energy, potential, l)
    C_inv = np.linalg.inv(C)
    
    R = basis @ C_inv @ basis

    return hbarc**2/(2*mu*channel_radius)*R


def S_matrix_lagrange(energy, N, potential, l=0):

    k=np.sqrt(2*mu*energy)/hbarc

    I = coulombH_minus( k * channel_radius, energy, l)
    I_diff = coulombH_minus( k * channel_radius, energy, l, d = True)

    R = R_matrix_lagrange(energy, N, potential, l)
    Z = I - R * k * channel_radius * I_diff

    S = 1/conj(Z) * Z
    
    return S

#==========================================
#Basis construction

def params_geometric(first_param, ratio, number_of_params):
    
    params = np.empty(number_of_params)
    
    params[0] = first_param
    for i in range(1,number_of_params):
        params[i] = first_param * ratio**(-i)

    return 1 / params**2


def gaussian(r, param, l):
    return r**(l+1) * np.exp( - param * r**2 )

#==========================================
#Hamiltonian matrix elements in Gaussian basis 

def gaussian_integral(a, param, k):
    return  gamma((k+1)/2) * gammainc((k+1)/2, param * a**2) / ( 2 * param**( (k+1)/2 ) )


def gaussian_integral_num(a, param, k):
    return quad(lambda r : gaussian(r, param, k-1), 0, a)[0]


def C_matrix(params, energy, l=0):
    
    dim = len(params)
    C = np.empty((dim,dim))

    for i in range(dim):
        for j in range(dim):
            
            overlap_elem = gaussian_integral( channel_radius, params[i] + params[j], 2*l + 2)
            
            kin_bloch_elem = ( 4 * params[i]*params[j] * gaussian_integral( channel_radius, params[i] + params[j], 2*l + 4 ) 
                           - 2*(l+1)*(params[i] + params[j])*gaussian_integral( channel_radius, params[i] + params[j], 2*l + 2 ) 
                           + (l+1)*(2*l+1)*gaussian_integral( channel_radius, params[i] + params[j], 2*l) )

            coulomb_elem = e2 * gaussian_integral( channel_radius, params[i] + params[j], 2*l+1)

            potential_elem = ( 200 * gaussian_integral (channel_radius, params[i] + params[j] + 1.487, 2*l+2)
                              - 91.85 * gaussian_integral (channel_radius, params[i] + params[j] + 0.465, 2*l+2))

            C[i,j] = hh * kin_bloch_elem + potential_elem + coulomb_elem - energy*overlap_elem

    return C

#============================================
#S-matrix calculation 

def R_matrix(params, energy, l=0):

    dim = len(params)

    R = np.empty((dim,dim))

    C = C_matrix(params, energy, l)
    C_inv = np.linalg.inv(C)

    basis = gaussian(channel_radius, params, l)
    
    R = basis @ C_inv @ basis

    return hbarc**2/(2*mu*channel_radius)*R


def S_matrix(params, energy, l=0):

    k=np.sqrt(2*mu*energy)/hbarc

    I = coulombH_minus( k * channel_radius, energy, l)
    I_diff = coulombH_minus( k * channel_radius, energy, l, d = True)

    R = R_matrix(params, energy, l)
    Z = I - R * k * channel_radius * I_diff

    S = 1/conj(Z) * Z
    
    return S


############################################################################
############################################################################
############################################################################
############################################################################

#Initialization of parameters - Gaussian basis 
params = params_geometric(0.2, 0.70, 12)

#Sample calculation and plot - Gaussian basis
start = time.time()
energies = np.arange(0.01,5,1)
phase_shifts = np.empty(len(energies))
for i in range(len(energies)):
    S = S_matrix(params, energies[i], l)
    print(R_matrix(params, energies[i], l))
    phase_shift = (log( S ) / (2j))
    phase_shifts[i] = mpmath.re(phase_shift)*180/np.pi
end = time.time()
print(f'Time = {end - start} s')

plt.ylabel('delta')
plt.xlabel('E_cm')
plt.plot(energies,phase_shifts,label='Gauss')

#Sample calculation and plot - Lagrange basis
N = 20 #number of Lagrange basis states

start = time.time()
phase_shifts = np.empty(len(energies))
for i in range(len(energies)):
    S = S_matrix_lagrange(energies[i], N, Minnesota_singlet, l)
    phase_shift = (log( S ) / (2j))
    phase_shifts[i] = mpmath.re(phase_shift)*180/np.pi
end = time.time()
print(f'Time = {end - start} s')

plt.ylabel('delta')
plt.xlabel('E_cm')
plt.plot(energies,phase_shifts,label='Lagrange')
plt.legend()
plt.show()







