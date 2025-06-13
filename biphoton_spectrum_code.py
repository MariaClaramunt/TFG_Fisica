import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm

c = 1
dc = c/2
c1 = c + dc
c2 = c - dc
L = 10/c

q_vals = np.linspace(-np.pi, np.pi, 200)
qs, qi = np.meshgrid(q_vals, q_vals)

# SSH HAMILTONIAN
def ssh_hamiltonian(q, c, dc):
    c1 = c + dc
    c2 = c - dc
    H = np.array([
        [0, c1 + c2 * np.exp(-1j * q)],
        [c1 + c2 * np.exp(1j * q), 0]
        ], dtype=complex)
    return H

# COMPUTATION OF THE SINGLE BAND EIGENVALUES
def eigenvalues(q_grid, c, dc):
    beta_plus = np.zeros_like(q_grid)
    beta_minus = np.zeros_like(q_grid)
    for i in range(q_grid.shape[0]):
        for j in range(q_grid.shape[1]):
            q = q_grid[i, j]
            H = ssh_hamiltonian(q, c, dc)
            eigvals = np.linalg.eigvalsh(H) # eigenvalues in ascending order and repeated according to multiplicity
            beta_minus[i, j], beta_plus[i, j] = eigvals
    return beta_plus, beta_minus

beta_plus_s, beta_minus_s = eigenvalues(qs, c, dc)
beta_plus_i, beta_minus_i = eigenvalues(qi, c, dc)

# COMPUTATION OF THE BIPHOTON SPECTRUM
beta_pp = beta_plus_s + beta_plus_i
beta_pm = beta_plus_s + beta_minus_i
beta_mp = beta_minus_s + beta_plus_i
beta_mm = beta_minus_s + beta_minus_i

# PLOTTING
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

vmin, vmax = -4*np.abs(c), 4*np.abs(c)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Colormap definition
from matplotlib.colors import LinearSegmentedColormap
colors = ["darkmagenta", "deeppink", "paleturquoise", "lightseagreen"]
nodes = [0.0, 0.49, 0.51, 1.0]
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

# Colormap unification
colors_pp = cmap(norm(beta_pp))
colors_pm = cmap(norm(beta_pm))
colors_mp = cmap(norm(beta_mp))
colors_mm = cmap(norm(beta_mm))

rstride=1
cstride=1
alpha=1
fontsize=14

surf1 = ax.plot_surface(qs, qi, beta_pp, facecolors=colors_pp, alpha=alpha, rstride=rstride, cstride=cstride)
surf2 = ax.plot_surface(qs, qi, beta_pm, facecolors=colors_pm, alpha=alpha, rstride=rstride, cstride=cstride)
surf3 = ax.plot_surface(qs, qi, beta_mp, facecolors=colors_mp, alpha=alpha, rstride=rstride, cstride=cstride)
surf4 = ax.plot_surface(qs, qi, beta_mm, facecolors=colors_mm, alpha=alpha, rstride=rstride, cstride=cstride)

ax.set_xlabel(r'$q_s$', fontsize=fontsize)
ax.set_ylabel(r'$q_i$', fontsize=fontsize)
ax.set_zlabel('')
ax.text(x=4.9, y=-4.9, z=-0.4, s=r'$\beta$', fontsize=fontsize, rotation=0)
ax.grid(False)
ax.set_box_aspect([1, 1, 0.9]) 

ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([ r'$-\pi$', r'$0$', r'$\pi$'], fontsize=fontsize)
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels([ r'$-\pi$', r'$0$', r'$\pi$'], fontsize=fontsize)
ax.set_zticks([-4, -2, 0, 2, 4])
ax.set_zticklabels([r"$-4|C|$", r"$-2|C|$", r"$0$", r"$2|C|$", r"$4|C|$"], fontsize=fontsize)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
cbar.set_ticks([-4, -2, 0, 2, 4])
cbar.set_ticklabels([r"$-4|C|$", r"$-2|C|$", r"$0$", r"$2|C|$", r"$4|C|$"], fontsize=fontsize)
cbar.set_label(r'$\beta$', fontsize=fontsize, rotation=0)
cbar.ax.yaxis.label.set_position((0, 0.55))

# Set view
ax.view_init(elev=10, azim=30)  

plt.show()