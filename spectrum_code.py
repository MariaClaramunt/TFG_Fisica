import numpy as np
import matplotlib.pyplot as plt

c=1
dc=-c/2
c1=c+dc
c2=c-dc
L=10/c

q_vals=np.linspace(0, 2*np.pi+0.2, 100)

# SSH HAMILTONIAN
def ssh_hamiltonian(q, c, dc):
    c1=c+dc
    c2=c-dc
    H = np.array([
        [0, c1 + c2 * np.exp(-1j * q)],
        [c1 + c2 * np.exp(1j * q), 0]
        ])
    eigenvals, eigenvecs = np.linalg.eigh(H)  # eigenvalues and eigenvectors in ascending order and repeated according to multiplicity
    return eigenvals, eigenvecs

# COMPUTATION OF THE EIGENVALUES
all_eigenvals = []
for q in q_vals:
    eigenvals, _ = ssh_hamiltonian(q, c, dc)
    all_eigenvals.append(eigenvals)
all_eigenvals = np.array(all_eigenvals) # list to array

# PLOTTING
fontsize= 30
linewidth = 3

plt.figure(figsize=(7,5))
plt.plot(q_vals, 2 * c * np.cos(q_vals / 2), color='blue', linewidth = linewidth) # dc=0 case
plt.plot(q_vals[q_vals>np.pi], all_eigenvals[:, 0][q_vals>np.pi], color='red', linewidth = linewidth)
plt.plot(q_vals[q_vals<np.pi], all_eigenvals[:, 1][q_vals<np.pi], color='red', linewidth = linewidth)

plt.xlabel(r'$q$', fontsize=fontsize)
ylabel=plt.ylabel(r'$\beta$', rotation=0,fontsize=fontsize)
ylabel.set_position((0, 0.42))
plt.xlim(0, 2*np.pi+0.2)
plt.ylim(-2, 2.2)
plt.xticks([ 0, np.pi, 2*np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize=fontsize)
plt.yticks([-2, -1, 0, 1, 2], [r'$-2|C|$', r'$-2|\delta C|$', r'$0$', r'$2|\delta C|$', r'$2|C|$'], fontsize=fontsize) #per c=1

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    plt.gca().spines[spine].set_linewidth(2)   
plt.gca().tick_params(width=2, length=8) 

plt.hlines(-1, 0, 2*np.pi, color='grey',linestyles='dashed', linewidth = linewidth)
plt.hlines(1, 0, 2*np.pi, color='grey',linestyles='dashed', linewidth = linewidth)

plt.tight_layout()
plt.show()