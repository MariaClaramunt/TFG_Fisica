import numpy as np
import matplotlib.pyplot as plt

c = 1
L = 10/c   
gamma = 1  
N = 21

# SSH Hamiltonian in real space
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

# Compute correlations
def correlation(dc, delta_beta_0):
        eigenvals, eigenvecs = ssh_real_space(N, c, dc)

        # PUMP PROFILE
        pump_profile = np.zeros(N)
        pump_profile[0] = 1  # Pumpimg the edge of the array
        
        psi = np.zeros((N, N), dtype=complex)
                
        for q_s in range(N):
            for q_i in range(N):
                # Compute coupling efficiency \Gamma(q_s, q_i)
                Gamma = 0
                for j in range(N):  # Sum over all waveguides
                    Gamma += pump_profile[j] * eigenvecs[j, q_s] * eigenvecs[j, q_i]
                
                # Phase mismatch \Delta\beta = \Delta\beta^{(0)} - beta(q_s) - \beta(q_i)
                Delta_beta = delta_beta_0 - eigenvals[q_s] - eigenvals[q_i]
                
                # Contribution to the biphoton state
                psi += ( gamma * Gamma * L * np.sinc(Delta_beta * L / (2 * np.pi)) * np.exp(-1j * Delta_beta * L / 2) * np.outer(eigenvecs[:, q_s], eigenvecs[:, q_i]))
        
        correlation_real = np.abs(psi)**2
        correlation_real /= np.max(correlation_real) # Normalize
        
        return correlation_real

# PLOTTING
fig, axs = plt.subplots(1, 3, figsize=(33, 7))
extent = [-1, 1, -4, 4]

fontsize=40
fontsize2=45

im0 = axs[0].imshow(correlation(0, 0), extent=[0, N-1, 0, N-1], origin='lower', cmap='jet')

axs[0].set_xlabel(r'$n_s$', fontsize=fontsize2)
axs[0].set_ylabel(r'$n_i$', rotation=0, fontsize=fontsize2)
axs[0].yaxis.set_label_coords(-0.2, 0.46)
axs[0].set_xticks([0, 5, 10, 15, 20], [r'$0$',r'$5$',r'$10$',r'$15$', r'$20$'], fontsize=fontsize)
axs[0].set_yticks([0, 5, 10, 15, 20], [r'$0$',r'$5$',r'$10$',r'$15$', r'$20$'], fontsize=fontsize)
axs[0].set_box_aspect(1)

axs[0].text(0.175, 1.1,'Homogeneous', transform=axs[0].transAxes, fontsize=fontsize2)


im1 = axs[1].imshow(correlation(c/2, 0), extent=[0, N-1, 0, N-1], origin='lower', cmap='jet')

axs[1].set_xlabel(r'$n_s$', fontsize=fontsize2)
axs[1].set_ylabel(r'$n_i$', rotation=0, fontsize=fontsize2)
axs[1].yaxis.set_label_coords(-0.2, 0.46)
axs[1].set_xticks([0, 5, 10, 15, 20], [r'$0$',r'$5$',r'$10$',r'$15$', r'$20$'], fontsize=fontsize)
axs[1].set_yticks([0, 5, 10, 15, 20], [r'$0$',r'$5$',r'$10$',r'$15$', r'$20$'], fontsize=fontsize)
axs[1].set_box_aspect(1)

axs[1].text(0.35, 1.1, 'Trivial', transform=axs[1].transAxes, fontsize=fontsize2)


im2 = axs[2].imshow(correlation(-c/2, 0), extent=[0, N-1, 0, N-1], origin='lower', cmap='jet')

axs[2].set_xlabel(r'$n_s$', fontsize=fontsize2)
axs[2].set_ylabel(r'$n_i$', rotation=0, fontsize=fontsize2)
axs[2].yaxis.set_label_coords(-0.2, 0.46)
axs[2].set_xticks([0, 5, 10, 15, 20], [r'$0$',r'$5$',r'$10$',r'$15$', r'$20$'], fontsize=fontsize)
axs[2].set_yticks([0, 5, 10, 15, 20], [r'$0$',r'$5$',r'$10$',r'$15$', r'$20$'], fontsize=fontsize)
axs[2].set_box_aspect(1)

axs[2].text(0.26, 1.1, 'Nontrivial', transform=axs[2].transAxes, fontsize=fontsize2)


# Unified colorbar
cbar=fig.colorbar(im0, ax=axs.ravel().tolist())
cbar.set_ticks([0.0,0.25, 0.5,0.75,1])
cbar.set_ticklabels([r"$0$",r"$0.25$",r"$0.5$",r"$0.75$",r"$1$"], fontsize=fontsize)
cbar.set_label(r'$|\psi(q_s,q_i)|^2$', fontsize=fontsize, rotation=0, labelpad=100)
cbar.ax.yaxis.label.set_position((0, 0.55))

#plt.tight_layout()
plt.show()