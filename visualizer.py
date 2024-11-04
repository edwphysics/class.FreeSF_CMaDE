import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX font rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')  # Set font to LaTeX default serif

# Plots Saving Folder
directory = './plots/'
report_dir = '/Users/edwinperez/Dropbox/MSc-Project/Report/mastersthesis/plots/CMaDE_SF/'

# Constants
m  = 1.e-22 	  # SF Mass in eV
T0 = 2.725e6  	  # CMB Temperature in microkelvins

# Load the data
data 	= np.loadtxt('output/lsfdm00_background.dat')
data_k0 = np.loadtxt('output/lsfdm00_perturbations_k0_s.dat')
data_cl = np.loadtxt('output/lsfdm00_cl.dat')
data_pk = np.loadtxt('output/lsfdm00_pk.dat')

# Extract Relevant Results
z 		= np.array(data[:,0])
H 		= np.array(data[:,3])
rho_g 	= np.array(data[:,8])
rho_b 	= np.array(data[:,9])
rho_l 	= np.array(data[:,11])
rho_nu 	= np.array(data[:,12])
rho_scf = np.array(data[:,14])
theta 	= np.array(data[:,17])
Omega_phi_scf = np.array(data[:,16])

# Perturbations Data
a_p 	   = np.array(data_k0[:,1])
delta_b    = np.array(data_k0[:,8])
delta0_scf = np.array(data_k0[:,20])
delta1_scf = np.array(data_k0[:,21])
delta_scf  = np.array(data_k0[:,17])

# MPS Data at z=0
k   = np.array(data_pk[:,0])
p_k = np.array(data_pk[:,1])

# Cl Multipole CMB
l 	  = np.array(data_cl[:,0])
cl_tt = np.array(data_cl[:,1])* T0**2

# Derived Parameters
a = 1./(1. + z)

# Scf Variables
x = np.sqrt(rho_scf)* np.sin(theta/2.)/ H # x: Kinetic Energy
y = np.sqrt(rho_scf)* np.cos(theta/2.)/ H # y: Potential Energy
Omega_scf = x**2 + y**2 			   	  # Omega_scf
kPhi = -np.sqrt(6)* H* y/m  			  # Scalar Field

# Density Parameters
Omega_g  = rho_g/H**2
Omega_b  = rho_b/H**2
Omega_l  = rho_l/H**2
Omega_nu = rho_nu/H**2

# Plot: kappaPhi
fig1 = plt.figure(figsize=(10,6)) 
ax1  = fig1.add_subplot(111) 
ax1.semilogx(a, kPhi, label=r'$\kappa\phi$', color='blue')
ax1.set_ylabel(r'$\kappa\phi(a)$', fontsize=20, fontweight='bold')
ax1.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=18)
#fig1.savefig(directory + 'kphi.pdf')
#fig1.savefig(report_dir + 'kphi.pdf')

# Plot: Parameter Densities
fig2 = plt.figure(figsize=(10,6)) 
ax2  = fig2.add_subplot(111) 
ax2.semilogx(a, Omega_scf, 'black', label=r"$\Omega_{\rm SFDM}$") # Dark Matter
ax2.semilogx(a, Omega_g, 'blue', label=r"$\Omega_{\gamma}$")      # Photons
ax2.semilogx(a, Omega_nu, 'orange', label=r"$\Omega_{\nu}$")      # Neutrinos
ax2.semilogx(a, Omega_b, 'red', label=r"$\Omega_b$")              # Baryons
ax2.semilogx(a, Omega_l, 'green', label=r"$\Omega_{\Lambda}$")    # Lambda
ax2.set_ylabel(r'$\Omega(a)$', fontsize=20, fontweight='bold')                         
ax2.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.legend(loc = 'upper left', fontsize = '12')
#fig2.savefig(directory + 'omegas.pdf')
#fig2.savefig(report_dir + 'omegas.pdf')

# Plot: Kinetic Energy
fig3 = plt.figure(figsize=(10,6)) 
ax3  = fig3.add_subplot(111) 
ax3.semilogx(a, x**2, label=r'$x^2$', color='blue')
ax3.set_ylabel(r'$x^2(a)$', fontsize=20, fontweight='bold')
ax3.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=18)
#fig3.savefig(directory + 'x2.pdf')
#fig3.savefig(report_dir + 'x2.pdf')

# Plot: Potential Energy
fig4 = plt.figure(figsize=(10,6)) 
ax4  = fig4.add_subplot(111) 
ax4.semilogx(a, y**2, label=r'$y^2$', color='blue')
ax4.set_ylabel(r'$y^2(a)$', fontsize=20, fontweight='bold')
ax4.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
ax4.tick_params(axis='both', which='major', labelsize=18)
#fig4.savefig(directory + 'y2.pdf')
#fig4.savefig(report_dir + 'y2.pdf')

# Plot: SFDM EoS
fig5 = plt.figure(figsize=(10,6)) 
ax5  = fig5.add_subplot(111) 
ax5.semilogx(a, -np.cos(theta), label=r'$w_\phi$', color='blue')
ax5.set_ylabel(r'$w_\phi(a)$', fontsize=20, fontweight='bold')
ax5.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
ax5.tick_params(axis='both', which='major', labelsize=18)
#fig5.savefig(directory + 'weff.pdf')
#fig5.savefig(report_dir + 'weff.pdf')

# Plot: Omega_phi_scf
fig6 = plt.figure(figsize=(10,6)) 
ax6  = fig6.add_subplot(111) 
ax6.semilogx(a, np.exp(Omega_phi_scf), label=r'$\Omega_\phi$', color='blue')
ax6.set_ylabel(r'$\Omega_\phi(a)$', fontsize=20, fontweight='bold')
ax6.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
ax6.tick_params(axis='both', which='major', labelsize=18)
#fig6.savefig(directory + 'omega_phi.pdf')
#fig6.savefig(report_dir + 'omega_phi.pdf')

# Plot: d_scf
fig7 = plt.figure(figsize=(10,6)) 
ax7  = fig7.add_subplot(111) 
ax7.plot(np.log10(a_p), np.log10(np.abs(delta0_scf)), label=r'$|\delta_\phi|$', color='black')
ax7.plot(np.log10(a_p), np.log10(np.abs(delta_b)), label=r'$|\delta_{\rm b}|$', color='blue')
ax7.set_ylabel(r'$\log{(|\delta|)}$', fontsize=20, fontweight='bold')
ax7.set_xlabel(r'$\log{(a)}$', fontsize=20, fontweight='bold')
ax7.tick_params(axis='both', which='major', labelsize=18)
ax7.legend(loc = 'upper left', fontsize = '12')
#fig7.savefig(directory + 'deltas.pdf')
#fig7.savefig(report_dir + 'deltas.pdf')

# Plot: C_lTT CMB Multipoles
fig8 = plt.figure(figsize=(10,6)) 
ax8  = fig8.add_subplot(111) 
ax8.plot(l, cl_tt, label=r'$\frac{l(l+2)C_l^{TT}}{2\pi}$', color='blue')
ax8.set_ylabel(r'$\frac{l(l+2)C_l^{TT}}{2\pi}[\mu K^2]$', fontsize=20, fontweight='bold')
ax8.set_xlabel(r'$l$', fontsize=20, fontweight='bold')
ax8.tick_params(axis='both', which='major', labelsize=18)
#fig8.savefig(directory + 'C_lTT.pdf')
#fig8.savefig(report_dir + 'C_lTT.pdf')

# Plot: MPS
fig9 = plt.figure(figsize=(10,6)) 
ax9  = fig9.add_subplot(111) 
ax9.plot(k, p_k, label=r'$P(k)$', color='blue')
ax9.set_xscale('log')
ax9.set_yscale('log')
ax9.set_ylabel(r'$P(k)[{\rm h^{-3}Mpc^{-3}}]$', fontsize=20, fontweight='bold')
ax9.set_xlabel(r'$k[{\rm h~Mpc^{-1}}]$', fontsize=20, fontweight='bold')
ax9.tick_params(axis='both', which='major', labelsize=18)
#fig9.savefig(directory + 'mps.pdf')
#fig9.savefig(report_dir + 'mps.pdf')

# Print Plots
plt.grid(True)
plt.show()