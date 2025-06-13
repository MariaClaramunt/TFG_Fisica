import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

c = 1

# INFINIE LATTICE

# SSH Hamiltonian
def ssh_hamiltonian(q, c1, c2):
    H = np.array([
        [0, c1 + c2 * np.exp(-1j * q)],
        [c1 + c2 * np.exp(1j * q), 0]
        ], dtype=complex)
    return H

# Computation of the eigenvalues
def eigenvalues(q_grid, c1, c2):
    beta_plus = np.zeros_like(q_grid)
    beta_minus = np.zeros_like(q_grid)
    for i in range(q_grid.shape[0]):
        for j in range(q_grid.shape[1]):
            q = q_grid[i, j]
            H = ssh_hamiltonian(q, c1, c2)
            eigvals = np.linalg.eigvalsh(H)
            beta_minus[i, j], beta_plus[i, j] = eigvals
    return beta_plus, beta_minus

q_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
qs, qi = np.meshgrid(q_vals, q_vals)

dcc_vals = np.linspace(-1, 1, 200)
beta_vals = np.linspace(-4, 4, 200)

# Computation of the density of states
dos_inf = np.zeros((len(beta_vals), len(dcc_vals)))

beta_edges = np.linspace(-4, 4, len(beta_vals) + 1)

for j, dcc in enumerate(dcc_vals):
    dc = dcc * c
    c1 = c + dc
    c2 = c - dc

    beta_plus_s, beta_minus_s = eigenvalues(qs, c1, c2)
    beta_plus_i, beta_minus_i = eigenvalues(qi, c1, c2)

    beta_pp = beta_plus_s + beta_plus_i
    beta_pm = beta_plus_s + beta_minus_i
    beta_mp = beta_minus_s + beta_plus_i
    beta_mm = beta_minus_s + beta_minus_i

    all_beta = np.concatenate([beta_pp.ravel(), beta_pm.ravel(), beta_mp.ravel(), beta_mm.ravel()])

    counts, bin_edges = np.histogram(all_beta, bins=beta_edges)

    dos_inf[:, j] = counts

# Normalization
dos_inf = dos_inf / dos_inf.max()


# FINITE LATTICE

# Computation of the density of states
def dos(N):

    # SSH Hamiltonian in real space
    def ssh_real_space(N, c1, c2):
        H = np.zeros((N, N), dtype=complex)
        for i in range(N - 1):
            if i % 2 == 0:
                H[i, i + 1] = c1
            else:
                H[i, i + 1] = c2
        H += H.conj().T  # Hermitian conjugate
        return H
    
    dcc_vals = np.linspace(-1, 1, 200)
    beta_vals = np.linspace(-4, 4, 200)
    
    dos = np.zeros((len(beta_vals), len(dcc_vals)))
    
    beta_edges = np.linspace(-4, 4, len(beta_vals) + 1)
    
    for j, dcc in enumerate(dcc_vals):
        dc = dcc * c
        c1 = c + dc
        c2 = c - dc
     
        H = ssh_real_space(N, c1, c2)
     
        single_photon_eigs = np.linalg.eigvalsh(H)
     
        biphoton_eigs = []
        for beta_s in single_photon_eigs:
            for beta_i in single_photon_eigs:
                biphoton_eigs.append(beta_s + beta_i)
     
        biphoton_eigs = np.array(biphoton_eigs)
     
        counts, _ = np.histogram(biphoton_eigs, bins=beta_edges)
        dos[:, j] = counts
    
    dos = dos / dos.max()
    return dos


# PLOTTING
fig, axs = plt.subplots(1, 3, figsize=(33, 7))

extent = [-1, 1, -4, 4]
norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)
fontsize = 30
linewidth = 4
cmap= 'turbo'

im0 = axs[0].imshow(dos_inf, extent=extent, origin='lower', aspect='auto', cmap=cmap, norm=norm)

axs[0].set_xlabel(r'$\delta C / C$', fontsize=fontsize)
ylabel0=axs[0].set_ylabel(r'$\beta$', rotation=0, fontsize=fontsize)
ylabel0.set_position((1, 0.46))
axs[0].set_xticks([-1, -1/3, 0, 1/3, 1])
axs[0].set_xticklabels([r'$-1$', r'$-1/3$', r'$0$', r'$1/3$', r'$1$'], fontsize=fontsize)
axs[0].set_yticks([-4, -2, 0, 2, 4])
axs[0].set_yticklabels([r'$-4|C|$', r'$-2|C|$', r'$0$', r'$2|C|$', r'$4|C|$'], fontsize=fontsize)
axs[0].set_box_aspect(1)

axs[0].text(-0.2, 1.1, '(a)', transform=axs[0].transAxes, fontsize=fontsize)

axs[0].vlines(-1/3, -4, 4, color='grey',linestyles='dashed', linewidth = linewidth)
axs[0].vlines(1/3, -4, 4, color='grey',linestyles='dashed', linewidth = linewidth)

im1 = axs[1].imshow(dos(100) , extent=extent, origin='lower', aspect='auto', cmap=cmap , norm=norm)

axs[1].set_xlabel(r'$\delta C / C$', fontsize=fontsize)
ylabel1=axs[1].set_ylabel(r'$\beta$', rotation=0, fontsize=fontsize)
ylabel1.set_position((1, 0.46))
axs[1].set_xticks([-1, -1/3, 0, 1/3, 1])
axs[1].set_xticklabels([r'$-1$', r'$-1/3$', r'$0$', r'$1/3$', r'$1$'], fontsize=fontsize)
axs[1].set_yticks([-4, -2, 0, 2, 4])
axs[1].set_yticklabels([r'$-4|C|$', r'$-2|C|$', r'$0$', r'$2|C|$', r'$4|C|$'], fontsize=fontsize)
axs[1].set_box_aspect(1)

axs[1].text(-0.2, 1.1, '(b)', transform=axs[1].transAxes, fontsize=fontsize)

axs[1].vlines(-1/3, -4, 4, color='grey',linestyles='dashed', linewidth = linewidth)
axs[1].vlines(1/3, -4, 4, color='grey',linestyles='dashed', linewidth = linewidth)

im2 = axs[2].imshow(dos(101) , extent=extent, origin='lower', aspect='auto', cmap=cmap ,norm=norm)

axs[2].set_xlabel(r'$\delta C / C$', fontsize=fontsize)
ylabel2=axs[2].set_ylabel(r'$\beta$', rotation=0, fontsize=fontsize)
ylabel2.set_position((1, 0.46))
axs[2].set_xticks([-1, -1/3, 0, 1/3, 1])
axs[2].set_xticklabels([r'$-1$', r'$-1/3$', r'$0$', r'$1/3$', r'$1$'], fontsize=fontsize)
axs[2].set_yticks([-4, -2, 0, 2, 4])
axs[2].set_yticklabels([r'$-4|C|$', r'$-2|C|$', r'$0$', r'$2|C|$', r'$4|C|$'], fontsize=fontsize)
axs[2].set_box_aspect(1)

axs[2].text(-0.2, 1.1, '(c)', transform=axs[2].transAxes, fontsize=fontsize)

axs[2].vlines(-1/3, -4, 4, color='grey',linestyles='dashed', linewidth = linewidth)
axs[2].vlines(1/3, -4, 4, color='grey',linestyles='dashed', linewidth = linewidth)

# Colorbar
cbar=fig.colorbar(im0, ax=axs.ravel().tolist())
cbar.set_ticks([0,0.01,0.1, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels([r"$0$",r"$0.01$", r"$0.1$",r"$0.25$",r"$0.5$",r"$0.75$",r"$1$"], fontsize=fontsize)
cbar.set_label(r'Density of states', fontsize=35, rotation=90)
cbar.ax.yaxis.label.set_position((0.4, 0.5))

plt.show()