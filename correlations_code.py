import numpy as np
import matplotlib.pyplot as plt

c = 1
L = 10/c
gamma = 1
E_p = [1/gamma, 1/gamma]

N = 211
delta_beta_0 = 3*c # First band excitation

# For second and third bands excitation CHANGE T0
# delta_beta_0 = 0 

# SSH Hamiltonian in real space and computation of the eigenvalues and eigenvectors
def ssh_real_space(N, c, dc):
    c1 = c + dc
    c2 = c - dc
    H = np.zeros((N, N), dtype=complex)
    for i in range(N - 1):
        if i % 2 == 0:
            H[i, i + 1] = c1
        else:
            H[i, i + 1] = c2
    H += H.conj().T  # Hermitian conjugate
    eigenvals, eigenvecs = np.linalg.eigh(H)
    return eigenvals, eigenvecs

# Compute the correlation function
def psi(dc, delta_beta_0, pump_pos):
    eigenvals, eigenvecs = ssh_real_space(N, c, dc)
    
    # DEFINITION OF THE PUMP PROFILE 
    pump_profile = np.zeros(N)

    # In-phase pumping of two adjacent waveguides at positions 'pump_pos' and 'pump_pos+1'
    pump_profile[pump_pos] = 1      
    pump_profile[pump_pos + 1] = 1
    
    # For out-of-phase pumping of two adjacent waveguides CHANGE TO
    # pump_profile[pump_pos] = 1      
    # pump_profile[pump_pos + 1] = -1
    
    # For in-phase pumping of two adjacent unit cells CHANGE TO
    # pump_profile[pump_pos] = 1      
    # pump_profile[pump_pos + 1] = 1
    # pump_profile[pump_pos + 2] = 1
    # pump_profile[pump_pos + 3] = 1 
    
    # For out-of-phase pumping of two adjacent unit cells CHANGE TO
    # pump_profile[pump_pos] = 1      
    # pump_profile[pump_pos + 1] = 1
    # pump_profile[pump_pos + 2] = -1
    # pump_profile[pump_pos + 3] = -1 
    
    psi = np.zeros((N, N), dtype=complex)
    
    for q_s in range(N):
        for q_i in range(N):
            # Compute the coupling efficiency \Gamma(q_s,q_i)
            Gamma = 0
            for j in range(N):  # Sum over all waveguides
                Gamma += pump_profile[j] * eigenvecs[j, q_s] * eigenvecs[j, q_i]
            
            # Phase mismatch \Delta\beta = \Delta\beta^{(0)} - beta(q_s) - \beta(q_i)
            Delta_beta = delta_beta_0 - eigenvals[q_s] - eigenvals[q_i]
            
            psi += (gamma * Gamma * L * np.sinc(Delta_beta * L / (2 * np.pi)) * np.exp(-1j * Delta_beta * L / 2) * np.outer(eigenvecs[:, q_s], eigenvecs[:, q_i]))

    return psi

# Set pump position in the bulk (center of the lattice)
pump_pos = N // 2 - 1

# COMPUTE CORRELATIONS IN REAL SPACE
# Trivial phase
real_trivial = np.abs(psi(c/2, delta_beta_0, pump_pos))**2
real_trivial /= np.max(real_trivial)  # Normalize

# Nontrivial phase
real_nontrivial = np.abs(psi(-c/2, delta_beta_0, pump_pos))**2
real_nontrivial /= np.max(real_nontrivial)  # Normalize

# COMPUTE CORRELATIONS IN FOURIER SPACE
# Trivial phase
psi_fourier_trivial = np.fft.fftshift(np.fft.fft2(psi(c/2, delta_beta_0, pump_pos)))
fourier_trivial = np.abs(psi_fourier_trivial)**2
fourier_trivial /= np.max(fourier_trivial)  # Normalize

# Get the center values
N_trivial = fourier_trivial.shape[0]
start_trivial = int(N_trivial / 4)
end_trivial = int(3 * N_trivial / 4)
fourier_trivial = fourier_trivial[start_trivial:end_trivial, start_trivial:end_trivial]

# Nontrivial phase
psi_fourier_nontrivial = np.fft.fftshift(np.fft.fft2(psi(-c/2, delta_beta_0, pump_pos)))
fourier_nontrivial = np.abs(psi_fourier_nontrivial)**2
fourier_nontrivial /= np.max(fourier_nontrivial)  # Normalize

# Get the center values
N_nontrivial = fourier_nontrivial.shape[0]  
start_nontrivial = int(N_nontrivial / 4)
end_nontrivial = int(3 * N_nontrivial / 4)
fourier_nontrivial = fourier_nontrivial[start_nontrivial:end_nontrivial, start_nontrivial:end_nontrivial]

# PLOTTING
fig, axs = plt.subplots(2, 2, figsize=(6.5,6))

vmin = 0
vmax = 1
fontsize = 18
position = (0,0.45)

im1 = axs[0, 0].imshow(real_trivial[int(N/2-13):int(N/2+15), int(N/2-13):int(N/2+15)], extent=[-N/2, N/2, -N/2, N/2], origin='lower', cmap='jet', vmin=vmin, vmax=vmax)

axs[0, 0].set_xticks([-N/2, -N/4, 0, N/4, N/2])
axs[0, 0].set_xticklabels([r'$-14$',r'$-7$', r'$0$', r'$7$', r'$14$'], fontsize=fontsize)
axs[0, 0].set_yticks([-N/2, -N/4, 0, N/4, N/2])
axs[0, 0].set_yticklabels([r'$-14$',r'$-7$', r'$0$', r'$7$', r'$14$'], fontsize=fontsize)
axs[0, 0].set_xlabel(r'$n_s$', fontsize=fontsize)
ylabel0=axs[0, 0].set_ylabel(r'$n_i$', rotation=0, fontsize=fontsize)
ylabel0.set_position(position)
axs[0, 0].text(0.3, 1.2, 'Trivial', transform=axs[0, 0].transAxes, fontsize=20, va='top', ha='left')
axs[0, 0].text(-0.6, 0.8, 'Real Space', rotation=90, transform=axs[0, 0].transAxes, fontsize=20, va='top', ha='left')


im2 = axs[0, 1].imshow(real_nontrivial[int(N/2-13):int(N/2+15), int(N/2-13):int(N/2+15)], extent=[-N/2, N/2, -N/2, N/2], origin='lower', cmap='jet', vmin=vmin, vmax=vmax)

axs[0, 1].set_xticks([-N/2, -N/4, 0, N/4, N/2])
axs[0, 1].set_xticklabels([r'$-14$',r'$-7$', r'$0$', r'$7$', r'$14$'], fontsize=fontsize)
axs[0, 1].set_yticks([-N/2, -N/4, 0, N/4, N/2])
axs[0, 1].set_yticklabels([r'$-14$',r'$-7$', r'$0$', r'$7$', r'$14$'], fontsize=fontsize)
axs[0, 1].set_xlabel(r'$n_s$', fontsize=fontsize)
ylabel1=axs[0, 1].set_ylabel(r'$n_i$', rotation=0, fontsize=fontsize)
ylabel1.set_position(position)
axs[0, 1].text(0.22, 1.2, 'Nontrivial', transform=axs[0, 1].transAxes, fontsize=20, va='top', ha='left')


im3 = axs[1, 0].imshow(fourier_trivial, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap='jet', vmin=vmin, vmax=vmax)

axs[1, 0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axs[1, 0].set_xticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$', r'$0$',r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
axs[1, 0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axs[1, 0].set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$', r'$0$',r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
axs[1, 0].set_xlabel(r'$q_s$', fontsize=fontsize)
ylabel2=axs[1, 0].set_ylabel(r'$q_i$', rotation=0, fontsize=fontsize)
ylabel2.set_position(position)
axs[1, 0].text(-0.6, 0.86, 'Fourier Space', rotation=90, transform=axs[1, 0].transAxes, fontsize=20, va='top', ha='left')


im4 = axs[1, 1].imshow(fourier_nontrivial, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap='jet', vmin=vmin, vmax=vmax)

axs[1, 1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axs[1, 1].set_xticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$', r'$0$',r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
axs[1, 1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axs[1, 1].set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$', r'$0$',r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
axs[1, 1].set_xlabel(r'$q_s$', fontsize=fontsize)
ylabel3=axs[1, 1].set_ylabel(r'$q_i$', rotation=0, fontsize=fontsize)
ylabel3.set_position(position)

# Unified colorbar
cbar_ax = fig.add_axes([1.05, 0.25, 0.02, 0.5])
cbar = fig.colorbar(im4, cax=cbar_ax, ticks=[0.0,0.25, 0.5,0.75,1])
cbar.set_ticklabels([r"$0$",r"$0.25$",r"$0.5$",r"$0.75$",r"$1$"], fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize, direction='out', length=5)
cbar.set_label(r'$|\psi(q_s,q_i)|^2$', fontsize=fontsize, rotation=0, labelpad=50)
cbar.ax.yaxis.label.set_position((0, 0.55))

plt.tight_layout()
plt.show()