import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# 1. PROBLEM 3: CARDIAC AP, [Ca2+] TRANSIENT, AND FORCE (A vs B)
# =============================================================================
t = np.linspace(0, 0.85, 1000)

# Action Potential A (Normal) and B (Reduced duration)
def cardiac_ap(t, plateau_end):
    vm = np.full_like(t, -83.0)
    upstroke = 1.0 / (1.0 + np.exp(-(t - 0.225) / 0.004))
    notch = 22.0 * np.exp(-np.maximum(0, t - 0.235) / 0.008) * (t >= 0.235)
    repol = 1.0 / (1.0 + np.exp((t - plateau_end) / 0.035))
    vm = -83.0 + upstroke * ((88.0 * repol) + notch)
    return vm

Vm_A = cardiac_ap(t, plateau_end=0.47)
Vm_B = cardiac_ap(t, plateau_end=0.33)

# Cytoplasmic [Ca2+] Transient
Ca_A = np.where(t > 0.22, ((t - 0.22) / 0.08)**2 * np.exp(-(t - 0.22) / 0.075), 0)
Ca_A = 0.1 + 1.4 * (Ca_A / np.max(Ca_A))

# Force Twitch A (Normal) and B (Reduced AP duration)
Force_A = np.where((t >= 0.22) & (t <= 0.70), np.sin(np.pi * (t - 0.22) / (0.70 - 0.22))**1.8, 0)
Force_B = np.where((t >= 0.22) & (t <= 0.52), 0.42 * np.sin(np.pi * (t - 0.22) / (0.52 - 0.22))**1.8, 0)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Subplot 1: Cardiac Action Potentials A & B with Ion Phase Labels
axes[0].plot(t, Vm_A, 'k-', lw=2.5, label='A (Normal AP)')
axes[0].plot(t, Vm_B, 'r--', lw=2.5, label='B (Reduced Duration)')
axes[0].set_title('Problem 3: Cardiac Action Potentials (A vs. B)', fontweight='bold')
axes[0].set_xlabel('Time (s)', fontweight='bold')
axes[0].set_ylabel('Vm (mV)', fontweight='bold')
axes[0].set_ylim(-100, 40)
axes[0].set_xlim(0, 0.85)
axes[0].annotate('Phase 0:\nNa+ Influx (In)', xy=(0.22, -20), xytext=(0.02, -10),
                 arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=8.5, fontweight='bold')
axes[0].annotate('Phase 1:\nBrief K+ Efflux', xy=(0.24, 18), xytext=(0.28, 25),
                 arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=8.5, fontweight='bold')
axes[0].annotate('Phase 2 (Plateau):\nCa2+ Influx = K+ Efflux', xy=(0.35, 3), xytext=(0.38, 12),
                 arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=8.5, fontweight='bold')
axes[0].annotate('Phase 3:\nK+ Efflux (Out)', xy=(0.50, -45), xytext=(0.56, -35),
                 arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=8.5, fontweight='bold')
axes[0].text(0.03, -78, 'Phase 4: K+ Efflux', fontsize=8.5, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, linestyle=':', alpha=0.6)

# Subplot 2: Cytoplasmic [Ca2+] vs Time
axes[1].plot(t, Ca_A, 'b-', lw=2.5)
axes[1].set_title('Problem 3 Warm-Up: Cytoplasmic [Ca2+] vs. Time', fontweight='bold')
axes[1].set_xlabel('Time (s)', fontweight='bold')
axes[1].set_ylabel('Cytoplasmic [Ca2+] (uM)', fontweight='bold')
axes[1].set_xlim(0, 0.85)
axes[1].set_ylim(0, 1.7)
axes[1].annotate('L-type Ca2+ influx +\nSR Ca2+ release (CICR)', xy=(0.27, 1.0), xytext=(0.02, 1.25),
                 arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=8.5, fontweight='bold')
axes[1].annotate('Removal via SERCA pump\n& Na+/Ca2+ exchanger', xy=(0.45, 0.7), xytext=(0.48, 1.05),
                 arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=8.5, fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.6)

# Subplot 3: Force vs Time (A vs B)
axes[2].plot(t, Force_A, 'k-', lw=3, label='A (Normal Force Twitch)')
axes[2].plot(t, Force_B, 'r--', lw=2.5, label='B (Reduced AP Duration)')
axes[2].set_title('Problem 3 Main: Predicted Force vs. Time', fontweight='bold')
axes[2].set_xlabel('Time (s)', fontweight='bold')
axes[2].set_ylabel('Force (relative units)', fontweight='bold')
axes[2].set_xlim(0, 0.85)
axes[2].set_ylim(-0.05, 1.2)
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('HW8_Problem3_Graphs.png', dpi=300)
plt.show()

# =============================================================================
# 2. PROBLEM 4: SKELETAL VS. CARDIAC SARCOMERE AT RESTING LENGTH
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5))

def draw_sarcomere(ax, z_left, z_right, thin_len, title, annotate_labels=False):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.0, 1.1)
    ax.axis('off')
    ax.set_title(title, fontweight='bold', fontsize=12)

    # Z-discs
    ax.plot([z_left, z_left], [-0.75, 0.75], 'k-', lw=5)
    ax.plot([z_right, z_right], [-0.75, 0.75], 'k-', lw=5)

    # Thick filament (Myosin) in center (-0.65 to +0.65)
    for y_thick in [-0.3, 0.3]:
        ax.plot([-0.65, 0.65], [y_thick, y_thick], color='#b91c1c', lw=6)
        # Myosin heads (leaving bare zone in the very center -0.1 to +0.1)
        for x_head in np.concatenate([np.linspace(-0.62, -0.12, 8), np.linspace(0.12, 0.62, 8)]):
            for dy in [-0.14, 0.14]:
                ax.plot([x_head, x_head], [y_thick, y_thick + dy], color='#b91c1c', lw=2)

    # Thin filaments (Actin) extending inward from Z-discs
    for y_thin in [-0.55, 0.0, 0.55]:
        # Left thin filament
        ax.plot([z_left, z_left + thin_len], [y_thin + 0.02, y_thin + 0.02], color='#1d4ed8', lw=3)
        # Right thin filament
        ax.plot([z_right, z_right - thin_len], [y_thin - 0.02, y_thin - 0.02], color='#1d4ed8', lw=3)

    if annotate_labels:
        ax.annotate('Z-disc', xy=(z_left, 0.75), xytext=(z_left - 0.25, 0.9),
                    arrowprops=dict(arrowstyle='->', lw=1.2), fontweight='bold')
        ax.annotate('Thin Filament (Actin)', xy=(z_left + 0.4, 0.57), xytext=(-0.6, 0.88),
                    arrowprops=dict(arrowstyle='->', lw=1.2), fontweight='bold', color='#1d4ed8')
        ax.annotate('Thick Filament (Myosin)', xy=(0.0, 0.3), xytext=(0.1, 0.88),
                    arrowprops=dict(arrowstyle='->', lw=1.2), fontweight='bold', color='#b91c1c')
        ax.annotate('Myosin Heads', xy=(0.45, 0.44), xytext=(0.65, 0.85),
                    arrowprops=dict(arrowstyle='->', lw=1.2), fontweight='bold', color='#b91c1c')

# Skeletal: resting length is optimal (L0 ~ 2.1 um), thin filaments don't cross center
draw_sarcomere(ax1, z_left=-1.1, z_right=1.1, thin_len=0.98,
               title='Skeletal Muscle Sarcomere at Resting Length (Optimal Length L0: Maximal Overlap, No Center Interference)',
               annotate_labels=True)

# Cardiac: resting length is shorter than L0 (~1.8 um), thin filaments double-overlap in center
draw_sarcomere(ax2, z_left=-0.82, z_right=0.82, thin_len=0.98,
               title='Ventricular Cardiac Sarcomere at Resting Length (< L0: Shorter Z-to-Z Distance, Thin Filaments Overlap in Center)',
               annotate_labels=False)
ax2.annotate('Double-overlap of thin filaments\nin center at resting length', xy=(0.0, 0.0), xytext=(-0.35, -0.92),
             arrowprops=dict(arrowstyle='->', lw=1.5), fontweight='bold', color='#1d4ed8')

plt.tight_layout()
plt.savefig('HW8_Problem4_Sarcomeres.png', dpi=300)
plt.show()