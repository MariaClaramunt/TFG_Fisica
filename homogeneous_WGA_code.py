import numpy as np
import matplotlib.pyplot as plt

N = 200  # Number of waveguides

c = 1
L = 10/c

q_vals = np.linspace(-np.pi, np.pi, N) 

# PUMP PROFILE
def A(kp):
    return np.ones_like(kp)

# CHANGE TO
# np.ones_like(kp) # to consider the pump coupled to the central waveguide
# 2* np.exp(-1j*kp/2)*np.cos(kp / 2) # to consider the in-phase pumping of two adjacent far from lattice edges
# 2j * np.exp(-1j*kp/2)*np.sin(kp / 2) # to consider the out-of-phase pumping of two adjacent far from lattice edges


# Compute the correlation function
psi_fourier = np.zeros((N, N), dtype=complex)

for i, qs in enumerate(q_vals):
    for j, qi in enumerate(q_vals):
        kp = qs + qi
        delta_beta = -2 * c * (np.cos(qs) + np.cos(qi))
        sinc_term = np.sinc(delta_beta * L / (2 * np.pi)) # np.sinc(x)=sin(np.pi x)/(np.pi x)
        phase = np.exp(-1j * delta_beta * L / 2)
        psi_fourier[i, j] = A(kp) * sinc_term * phase
   
# Compute correlation in Fourier space
fourier= np.abs(psi_fourier)**2
fourier=fourier/fourier.max() # Normalize

# Compute correlation in real space
psi_real = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(psi_fourier)))
real = np.abs(psi_real)**2
real=real/real.max()

# PLOTTING
fig=plt.figure(figsize=(7, 6))

cmap= 'jet'
fontsize =18

plt.subplot(2,2,1)
im1=plt.imshow(fourier, extent= [-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap=cmap, vmin=0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$',r'$-\frac{\pi}{2}$', r'$0$',r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$',r'$-\frac{\pi}{2}$', r'$0$',r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)

plt.xlabel(r'$q_s$', fontsize=fontsize)
plt.ylabel(r'$q_i$', fontsize=fontsize, rotation=0)

plt.subplot(2,2,2)
im2=plt.imshow(real[int(N/2-21):int(N/2+22),int(N/2-21):int(N/2+22)], extent= [-N/2, N/2, -N/2, N/2], origin='lower', cmap=cmap, vmin=0)
plt.xticks([-N/2,-N/4, 0, N/4, N/2], [r'$-20$',r'$-10$', r'$0$',r'$10$', r'$20$'], fontsize=fontsize)
plt.yticks([-N/2,-N/4, 0, N/4, N/2], [r'$-20$',r'$-10$', r'$0$',r'$10$', r'$20$'], fontsize=fontsize)

plt.xlabel(r'$n_s$', fontsize=fontsize)
plt.ylabel(r'$n_i$', fontsize=fontsize, rotation=0)

# Unified colorbar
cbar_ax = fig.add_axes([1, 0.56, 0.015, 0.4])
cbar = fig.colorbar(im2, cax=cbar_ax, ticks=[0.0,0.25,0.5,0.75,1])
cbar.set_ticklabels([r"$0$",r"$0.25$",r"$0.5$",r"$0.75$",r"$1$"], fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize, direction='out', length=5)
cbar.set_label(r'$|\psi(q_s,q_i)|^2$', fontsize=fontsize, rotation=0, labelpad=50)

plt.tight_layout()

plt.show()