import numpy as np
import mpmath
import time
import matplotlib
import matplotlib.pyplot as plt
from mpmath import coulombf, coulombg, log, conj
from scipy.special import gamma, gammainc, factorial 
from scipy.integrate import quad
from numpy.linalg import inv 
from itertools import permutations
from scipy.linalg import eigh

#==========================================
#Coulomb functions

def coulombF(r,energy,l):
    #regular coulomb function
    k = np.sqrt(2*mu_tp*energy)
    eta = e2*mu_tp/(k*hbarc)

    return coulombf(l,eta,r)


def coulombG(r,energy,l):
    #irregular coulomb function
    k = np.sqrt(2*mu_tp*energy)
    eta = e2*mu_tp/(k*hbarc)

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
#Basis construction - realative coordinate 

def basis_params(file):
    params = np.loadtxt(file)[:,1:]
    return params


def gaussian(r, param, l=0):
    return r**(l+1)*np.exp( - 0.5 * param * r**2 )


def params_geometric(first_param, ratio, number_of_params):
    
    params = np.empty(number_of_params)
    
    params[0] = first_param
    for i in range(1,number_of_params):
        params[i] = first_param * ratio**(-i)

    return 1 / params**2

#==========================================
#Jacobi coordinates

def U_matrix(npart):
    #jacobi transformation matrix 
    
    U = np.zeros((npart,npart))
    
    masses_sum = mass[0]
    
    U[0,0] = 1
    U[0,1] = -1
    
    for i in range(1, npart):
        masses_sum += mass[i]
        for j in range(0, i+1):
            U[i,j] = mass[j] / masses_sum

    for i in range(1, npart-1):
        U[i,i+1] = -1

    return U


def lambda_matrix(U):
    # lambda matrix, (2.11) in Suzuki Varga
    npart = len(U[:,0])
    L = np.zeros((npart-1, npart-1))
    
    for i in range(npart-1):
        L[i,i] = np.sum( U[i,:] * U[i,:] / mass )

    return L


def w_vector(U):
    #aux. w vector, (2.13) in Suzuki Varga
    npart = len(U[:,0])
    w = np.zeros((npart, npart, npart-1))

    U_inv = inv(U)
    for i in range(npart):
        for j in range(npart):
            w[i,j,:] =  U_inv[i,:-1] - U_inv[j,:-1] 

    return w

#==========================================
#Antisymmetrization 

def permutation_matrices(npart, U):
    #array of perm. matrices for every possible permutation of Jacobi coordinates 
    
    npermut = int(factorial(npart))
    P = np.zeros((npermut, npart-1, npart-1))
    
    particle_permuts = np.array(list(permutations([i for i in range(npart)])))

    for k in range(npermut):
        C = np.zeros((npart, npart))
        for i in range(npart):
            for j in range(npart):
                if j == particle_permuts[k, i]: C[i,j] = 1 
        #print(C)
        P[k,:,:] = (U @ C @ inv(U))[:-1,:-1]

    return P

def permutation_signs(npart):
    #signs of permutations of particles, sign = (-1)^(# of inversions)

    npermut = int(factorial(npart))
    particle_permuts = np.array(list(permutations([i for i in range(npart)])))
    signs = np.zeros(npermut)
    
    for k in range(npermut):
        transitions = 0
        for i in range(npart):
            j = i + 1

            for j in range(j, npart):
                if particle_permuts[k,i] > particle_permuts[k,j]: 
                    transitions = transitions + 1
        
        signs[k] = (-1)**transitions
    
    return signs

#==========================================
#Spin/isospin elements

def spin_iso_elems_identity(npart, permut_particles, permutation_signs):
    #spin and isospin identity matrix elements 
    npermut = int(factorial(npart))
    elems = np.zeros(npermut)

    for k in range(npermut):

        iso_elem = 0
        #isospin 
        for term1 in range(nisc): 
            for term2 in range(nisc):
                
                elem = cisc[term1] * cisc[term2]

                for i in range(npart):
                
                    if iso[term1, permut_particles[k, i]] != iso[term2, i]:
                        elem *= 0
                    else:
                        elem *= 1

                iso_elem += elem 

        spin_elem = 0 
        #spin 
        for term1 in range(nspc): 
            for term2 in range(nspc):
                
                elem = cspc[term1] * cspc[term2]

                for i in range(npart):

                    if isp[term1, permut_particles[k, i]] != isp[term2, i]:
                        elem *= 0
                    else:
                        elem *= 1

                spin_elem += elem 

        elems[k] = spin_elem * iso_elem * permutation_signs[k]
    
    return np.sign(elems) #all elems normalized to 1 or 0 


def spin_iso_elems_2b(npart, permut_particles, permut_signs):
    #Two-body spin-isospin matrix elements WIGNER
    npairs = int(npart*(npart-1)/2)
    npermut = int(factorial(npart))
    spiso_elem_two_body = np.zeros((5,npermut,npairs))

    for permut in range(0,npermut) :
       
        pair=0
        
        for part1 in range(0,npart) :
            for part2 in range(part1+1,npart) :
        
                isospin_elem_id_two_body=0
                isospin_elem_Pt_two_body=0
                isospin_elem_pp_two_body=0
            
                for isos_term1 in range(0,nisc) :
                    for isos_term2 in range(0,nisc) :
            
                        aux=cisc[isos_term1]*cisc[isos_term2]
                
                        for part in range(0,npart) :
                            if part==part1 or part==part2 :
                                aux*=1.
                            else :                   
                                if iso[isos_term1,permut_particles[permut,part]] != iso[isos_term2,part] :
                                    aux*=0
                                else :
                                    aux*=1.
                                    
                        #Isospin identity id
                        if (iso[isos_term1,permut_particles[permut,part1]] == iso[isos_term2,part1]) and \
                           (iso[isos_term1,permut_particles[permut,part2]] == iso[isos_term2,part2]) :
                           
                            isospin_elem_id_two_body+=aux            
                            
                        #Isospin exchange Pt
                        if (iso[isos_term1,permut_particles[permut,part1]] == iso[isos_term2,part2]) and \
                           (iso[isos_term1,permut_particles[permut,part2]] == iso[isos_term2,part1]) :
                           
                            isospin_elem_Pt_two_body+=aux            
                            
                        #proton-proton pair (Coulomb)
                        if (iso[isos_term1,permut_particles[permut,part1]] == 1) and (iso[isos_term2,part1]==1) and \
                           (iso[isos_term1,permut_particles[permut,part2]] == 1) and (iso[isos_term2,part2]==1) :
                            isospin_elem_pp_two_body+=aux            
            
                spin_elem_id_two_body=0
                spin_elem_Ps_two_body=0

                for spin_term1 in range(0,nspc) :
                    for spin_term2 in range(0,nspc) :
            
                        aux=cspc[spin_term1]*cspc[spin_term2]
                
                        for part in range(0,npart) :
                            if part==part1 or part==part2 :
                                aux*=1.
                            else :                   
                                if isp[spin_term1,permut_particles[permut,part]] != isp[spin_term2,part] :
                                    aux*=0
                                else :
                                    aux*=1.
                                    
                        #Isospin identity id
                        if (isp[spin_term1,permut_particles[permut,part1]] == isp[spin_term2,part1]) and \
                           (isp[spin_term1,permut_particles[permut,part2]] == isp[spin_term2,part2]) :
                           
                            spin_elem_id_two_body+=aux            
                            
                        #Isospin exchange Pt
                        if (isp[spin_term1,permut_particles[permut,part1]] == isp[spin_term2,part2]) and \
                           (isp[spin_term1,permut_particles[permut,part2]] == isp[spin_term2,part1]) :
                           
                            spin_elem_Ps_two_body+=aux             
        

                #Two-body spin-isospin matrix elements WIGNER        
                spiso_elem_two_body[0,permut,pair] = isospin_elem_id_two_body * spin_elem_id_two_body * permut_signs[permut] 

                #Two-body spin-isospin matrix elements MAJORANA
                spiso_elem_two_body[1,permut,pair] = -isospin_elem_Pt_two_body * spin_elem_Ps_two_body * permut_signs[permut]
            
                #Two-body spin-isospin matrix elements BARTLETT
                spiso_elem_two_body[2,permut,pair] = isospin_elem_id_two_body * spin_elem_Ps_two_body * permut_signs[permut]
            
                #Two-body spin-isospin matrix elements HAISENBERG
                spiso_elem_two_body[3,permut,pair] = -isospin_elem_Pt_two_body * spin_elem_id_two_body * permut_signs[permut]
            
                #Two-body spin-isospin matrix elements COULOMB
                spiso_elem_two_body[4,permut,pair] = isospin_elem_pp_two_body * spin_elem_id_two_body * permut_signs[permut]

                pair+=1
    
    return np.sign(spiso_elem_two_body) #all elems normalized to 1 or 0 

#==========================================
#Hamiltonian matrix elements in Gaussian basis, full space minus external part

def hamiltonian_elems_full(params1, params2, U, P, spiso_elems_id, spiso_elems_2b):
    #hamiltonian matrix elems in full space
    npermut = int(factorial(npart))

    lambd = lambda_matrix(U)
    w = w_vector(U)

    overlap = 0
    kin_ener = 0
    potential = 0

    for permut in range(npermut):

        A = params1
        A_permut = np.transpose(P[permut]) @ params2 @ P[permut]
        detB = np.linalg.det(A + A_permut)

        #overlap elem 
        overlap += ( 2*np.pi / detB )**1.5 * spiso_elems_id[permut]

        #kinetic energy elem 
        kin_ener += hbarc**2 * 1.5 * np.trace( A @ inv(A + A_permut) @ A_permut @ lambd) * (2*np.pi / detB)**1.5 * spiso_elems_id[permut]

        #potential + coulomb elem
        for i in range(npart-1):
            for j in range(i+1, npart):
    
                c_inv = w[i,j] @ inv(A + A_permut) @ np.transpose(w[i,j]) 
                c = 1 / c_inv
                
                #coulomb 
                elem = np.sqrt(0.5 * c) * spiso_elems_2b[4,permut,i] * (2*np.pi / detB)**1.5   
                potential += z1*z2*e2 * 2/(np.pi)**0.5 * elem 

                #potential
                for op_term in range(4):
                    for term in range(npt):
                        c_inv = w[i,j] @ inv(A + A_permut) @ np.transpose(w[i,j]) 
                        c = 1 / c_inv
                        potential += vpot[op_term, term] * ( c/(2*apot[op_term, term] + c) ) **1.5 * spiso_elems_2b[op_term,permut,i] * (2*np.pi / detB)**1.5

    return np.array([overlap, kin_ener, potential])#/npermut


def energy_eigvals(hamiltonian_elems, eigvectors = False):
    #energy spectrum of the hamiltonian 
    h_matrix = hamiltonian_elems[:,:,1]+hamiltonian_elems[:,:,2]
    n_matrix = hamiltonian_elems[:,:,0]
    
    eigvals, eigvecs = eigh(h_matrix, n_matrix, eigvals_only = False)

    if eigvectors == False: return eigvals
    else: return eigvals, eigvecs


def overlap_ext_elem(params1, params2):
    #overlap matrix element in external region 
    a = channel_radius
    param = params1 + params2
    
    return 4*np.pi*quad(lambda r : r*gaussian(r, param), a, np.inf)[0]


def kin_ener_ext_elem(params1, params2):
    #kinetic energy matrix element in external region 
    a = channel_radius
    param = params1 + params2

    return - hh_tp * 4*np.pi*quad(lambda r : gaussian(r, param)*(params2**2 * r**3 - 3 * params2 * r), a, np.inf)[0]


def coulomb_ext_elem(params1, params2):
    #coulomb matrix element in external region 
    a = channel_radius
    param = params1 + params2
    
    return 4*np.pi*e2*quad(lambda r :  gaussian(r, param), a, np.inf)[0]


def bloch_elem(params1, params2):
    #bloch operator matrix element
    a = channel_radius
    param = params1 + params2

    return   4*np.pi * hh_tp * np.exp( - 0.5 * param * a**2 ) * (1 - params2 * a**2) * a


def C_matrix_SVM(params_rel, params, energy, l=0):

    dim = len(params_rel)
    C = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(dim):
            
            overlap_elem = elems_pd[i,j,0] - overlap_ext_elem(params_rel[i], params_rel[j]) 

            kin_elem = elems_pd[i,j,1] - kin_ener_ext_elem(params_rel[i], params_rel[j])

            potential_elem = elems_pd[i,j,2] - coulomb_ext_elem(params_rel[i], params_rel[j])
 
            bloch = bloch_elem(params_rel[i], params_rel[j])

            C[i,j] = kin_elem + bloch + potential_elem - (energy+energy_deuteron)*overlap_elem  - energy_deuteron*overlap_ext_elem(params_rel[i], params_rel[j]) 

    return C

#============================================
#S-matrix calculation 

def R_matrix_SVM(params_rel, params, energy, l=0):

    dim = len(params_rel)

    R = np.zeros((dim,dim))

    C = C_matrix_SVM(params_rel, params, energy)
    C_inv = inv(C)

    basis = np.exp( - 0.5 * params_rel * channel_radius**2 )
    
    R = basis @ C_inv @ basis 

    return 4*np.pi*hh_tp*channel_radius*R


def S_matrix_SVM(params_rel, params, energy, l=0):

    k=np.sqrt(2*mu_tp*energy)/hbarc

    I = coulombH_minus( k * channel_radius, energy, l)
    I_diff = coulombH_minus( k * channel_radius, energy, l, d = True)

    R = R_matrix_SVM(params_rel, params, energy, l)
    Z = I - R * k * channel_radius * I_diff
    #print(f'Z = {Z}')
    S = 1/conj(Z) * Z
    
    return S

############################################################################
############################################################################

#==========================================
#Physical constants and parameters 

hbarc = 197.3269804
m = 938.918
mu = m/2
hh = hbarc**2 / (2*mu)
e2 = 0#1.44 
z1 = 1
z2 = 1
channel_radius = 7 #fm
l = 0 #orbital momentum 

m_target = 2*m 
m_projectile = m
mu_tp = m_target * m_projectile / (m_target + m_projectile)
hh_tp = hbarc**2 / (2*mu_tp)

#============= Potential terms =============
npt = 2

vpot = np.zeros ((4,npt))
apot = np.zeros ((4,npt))

#MN spin triplet
vpot[0,0]= 200.000
apot[0,0]= 1.487 

vpot[0,1]= -178
apot[0,1]= 0.639


'''
#Wigner terms 
vpot[0,0] = 100.000
apot[0,0] = 1.487 

vpot[0,1] = -44.50
apot[0,1] = 0.639

vpot[0,2] = -22.9625
apot[0,2] = 0.465

#Majorana terms 
vpot[1,0]= 100.000 
apot[1,0]=1.487

vpot[1,1]= -44.50
apot[1,1]=0.639

vpot[1,2]= -22.9625
apot[1,2]=0.465

#Bartlett terms 
vpot[2,0]=   0.000
apot[2,0]=1.487

vpot[2,1]= -44.50
apot[2,1]=0.639

vpot[2,2]= +22.9625
apot[2,2]=0.465

#Haisenberg terms
vpot[3,0]=   0.000
apot[3,0]=1.487

vpot[3,1]= -44.50   
apot[3,1]=0.639 

vpot[3,2]= +22.9625
apot[3,2]=0.465
'''

#=================== Basis =================
npart = 2

file = 'd_Minnesota_SU4.res'
params_SVM = basis_params(file)[:10] #deuteron parameters, calculated by SVM
params_geo = params_geometric(0.2, 0.7, 15) #rel. coordinate basis parameters 
#print(f'Rel. basis: {1/(params_geo)**0.5}')
#print(1/(params_SVM)**0.5)

A = np.zeros((len(params_SVM[:,0]), npart-1, npart -1))

for k in range(len(params_SVM[:,0])):
    A[k,:,:] = params_SVM[k,0:]

#======= Initialization - Deuteron ==========
#Masses 
mass = np.full(npart, m)

#Spin 
nspc = 1

cspc = np.zeros(nspc)
isp  = np.zeros((nspc,npart))

cspc[0]=1.

isp[0,0]=2
isp[0,1]=2

#Isospin 
nisc = 2

cisc = np.zeros(nisc)
iso  = np.zeros((nisc,npart))

cisc[0]=1.

iso[0,0]=1
iso[0,1]=2

cisc[1]=-1.

iso[1,0]=2
iso[1,1]=1

#U matrix
U = U_matrix(npart)

#permutation matrices and signs 
permut_particles = np.array(list(permutations([i for i in range(npart)])))
signs = permutation_signs(npart)
P = permutation_matrices(npart, U)

#spin/isospin elements 
spiso_elems_identity = spin_iso_elems_identity(npart, permut_particles, signs)
spiso_elems_2b = spin_iso_elems_2b(npart, permut_particles, signs)

#Hamiltonian elements  
elems_deuteron = np.zeros( (len(A[:,0,0]), len(A[:,0,0]), 3) )
elems_deuteron_full = np.zeros(3)

for i in range(len(A[:,0,0])):
    for j in range(len(A[:,0,0])):
        elems_deuteron[i,j,:] = hamiltonian_elems_full(A[i],A[j],U,P,spiso_elems_identity,spiso_elems_2b)
    
eigvecs_deuteron = energy_eigvals(elems_deuteron, eigvectors = True)[1][:,0]
energy_deuteron = energy_eigvals(elems_deuteron)[0]
print(f'Deuteron energy = {energy_deuteron}')

for i in range(len(A[:,0,0])):
    for j in range(len(A[:,0,0])):
        elems_deuteron_full[:] += elems_deuteron[i,j,:]*eigvecs_deuteron[i]*eigvecs_deuteron[j]

#======= Initialization - pd ==========
#Masses 
npart = 3
mass = np.full(npart, m)

#Isospin
nisc = 1

cisc = np.zeros(nisc)
iso  = np.zeros((nisc,npart))

cisc[0]=1.


iso[0,0]=1
iso[0,1]=2
iso[0,2]=3

#cisc[1]=-1.

#iso[1,0]=2
#iso[1,1]=1
#iso[1,2]=1

#Spin 
nspc = 1

cspc = np.zeros(nspc)
isp  = np.zeros((nspc,npart))

cspc[0]=1.

isp[0,0]=1
isp[0,1]=1
isp[0,2]=1

#U matrix
U = U_matrix(npart)

#permutation matrices and signs 
permut_particles = np.array(list(permutations([i for i in range(npart)])))
signs = permutation_signs(npart)
P = permutation_matrices(npart, U)

#spin/isospin elements 
spiso_elems_identity = spin_iso_elems_identity(npart, permut_particles, signs)
spiso_elems_2b = spin_iso_elems_2b(npart, permut_particles, signs)

#pd matrix elements 
elems_pd = np.zeros((len(params_geo), len(params_geo),3))

print('Calculating pd matrix elems...')

for i in range(len(params_geo)):
    for j in range(len(params_geo)):
        for k in range(len(eigvecs_deuteron)):
            for l in range(len(eigvecs_deuteron)):
                params1 = np.array([A[k,0,0],0,0,params_geo[i]]).reshape((2,2))
                params2 = np.array([A[l,0,0],0,0,params_geo[j]]).reshape((2,2))
            
                elems_pd[i,j,:] += hamiltonian_elems_full(params1, params2, U, P, spiso_elems_identity, spiso_elems_2b)*eigvecs_deuteron[k]*eigvecs_deuteron[l]       
            
print('..done')

###################################################
################# Calculation #####################
###################################################


#Phase shifts as a function of channel radius, constant energy
l = 0
energy = 1
for i in range(2,31):
    channel_radius = i
    S = S_matrix_SVM(params_geo, A, energy)
    plt.scatter(i, mpmath.re(log( S  ) / (2j))*180/np.pi, color='C0')
plt.show()

l = 0

##################################################
#Phase shifts as a function of energy, constant radius 
channel_radius = 6 #fm


energies = np.arange(0.1,10,0.05)
phase_shifts = np.zeros(len(energies))
for i in range(len(energies)):
    S = S_matrix_SVM(params_geo, A, energies[i])
    phase_shift = (log( S  ) / (2j))
    phase_shifts[i] = mpmath.re(phase_shift)*180/np.pi
    #if phase_shifts[i] > 0: phase_shifts[i] = phase_shifts[i]-180
    print(f'E = {energies[i]:.2f}  Delta = {phase_shifts[i]:.2f}')
end = time.time()
plt.plot(energies,phase_shifts,ls='-',label='SVM elems')

plt.legend()
plt.show()


