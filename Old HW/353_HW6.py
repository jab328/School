import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf_filename = "Homework_6_Plots.pdf"

def plot_phase_portrait(ax, A, title, eigvecs=None):
    """Helper to plot direction field and trajectories for x' = A x."""
    x1 = np.linspace(-3, 3, 20)
    x2 = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1, x2)
    U = A[0, 0] * X1 + A[0, 1] * X2
    V = A[1, 0] * X1 + A[1, 1] * X2
    
    # Normalize arrows for clean direction field
    N = np.sqrt(U**2 + V**2)
    N[N == 0] = 1
    ax.quiver(X1, X2, U/N, V/N, color='gray', alpha=0.5, pivot='mid')
    
    # Plot trajectories using streamplot
    x1_dense = np.linspace(-3, 3, 100)
    x2_dense = np.linspace(-3, 3, 100)
    X1_d, X2_d = np.meshgrid(x1_dense, x2_dense)
    U_d = A[0, 0] * X1_d + A[0, 1] * X2_d
    V_d = A[1, 0] * X1_d + A[1, 1] * X2_d
    ax.streamplot(X1_d, X2_d, U_d, V_d, color='royalblue', linewidth=1.2, density=0.8)
    
    # Plot eigenvectors if provided (for real distinct eigenvalues)
    if eigvecs is not None:
        s = np.linspace(-3, 3, 100)
        for vec, label in eigvecs:
            ax.plot(s * vec[0], s * vec[1], 'r--', lw=1.5, label=label)
        ax.legend(loc='upper right', fontsize=8)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(True, linestyle=':', alpha=0.5)

with PdfPages(pdf_filename) as pdf:

    # =========================================================================
    # PAGE 1: Section 6.4 (Problems 1, 2, 13)
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle("Section 6.4: Discontinuous Forcing Functions", fontsize=14, fontweight='bold')

    # --- Problem 1 ---
    t1 = np.linspace(0, 6 * np.pi, 1000)
    u_3pi = np.where(t1 >= 3 * np.pi, 1.0, 0.0)
    f1 = 1.0 - u_3pi
    y1 = np.sin(t1) + (1.0 - np.cos(t1)) - u_3pi * (1.0 - np.cos(t1 - 3 * np.pi))

    axes[0, 0].plot(t1, f1, 'r-', lw=2)
    axes[0, 0].set_title("6.4 #1(a): Forcing Function f(t)")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("f(t)")
    axes[0, 0].set_ylim(-0.2, 1.4)
    axes[0, 0].grid(True, linestyle=':')

    axes[0, 1].plot(t1, y1, 'b-', lw=2, label="y(t)")
    axes[0, 1].plot(t1, f1, 'r--', alpha=0.5, label="f(t)")
    axes[0, 1].set_title("6.4 #1(c): Solution y(t)")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("y(t)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':')

    # --- Problem 2 ---
    t2 = np.linspace(0, 4 * np.pi, 1000)
    u_pi = np.where(t2 >= np.pi, 1.0, 0.0)
    u_2pi = np.where(t2 >= 2 * np.pi, 1.0, 0.0)
    h2 = u_pi - u_2pi
    
    def g_func(tau):
        return 0.5 * (1.0 - np.exp(-tau) * (np.cos(tau) + np.sin(tau)))
    
    y2 = (np.exp(-t2) * np.sin(t2) 
          + u_pi * g_func(np.maximum(0, t2 - np.pi)) 
          - u_2pi * g_func(np.maximum(0, t2 - 2 * np.pi)))

    axes[1, 0].plot(t2, h2, 'r-', lw=2)
    axes[1, 0].set_title("6.4 #2(a): Forcing Function h(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("h(t)")
    axes[1, 0].set_ylim(-0.2, 1.4)
    axes[1, 0].grid(True, linestyle=':')

    axes[1, 1].plot(t2, y2, 'b-', lw=2, label="y(t)")
    axes[1, 1].plot(t2, 0.5 * h2, 'r--', alpha=0.5, label="Equilibrium h(t)/2")
    axes[1, 1].set_title("6.4 #2(c): Solution y(t)")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("y(t)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':')

    # --- Problem 13 ---
    t13_a = np.linspace(0, 6 * np.pi, 1000)
    f13_a = np.ones_like(t13_a)
    for k in range(1, 7):
        f13_a += 2 * ((-1)**k) * np.where(t13_a >= k * np.pi, 1.0, 0.0)

    axes[2, 0].plot(t13_a, f13_a, 'r-', lw=2)
    axes[2, 0].set_title("6.4 #13(a): f(t) on [0, 6π]")
    axes[2, 0].set_xlabel("t")
    axes[2, 0].set_ylabel("f(t)")
    axes[2, 0].set_ylim(-1.5, 1.5)
    axes[2, 0].grid(True, linestyle=':')

    t13_c = np.linspace(0, 60, 2000)
    y13_c = 1.0 - np.cos(t13_c)
    for k in range(1, 16):
        uk = np.where(t13_c >= k * np.pi, 1.0, 0.0)
        y13_c += 2 * ((-1)**k) * uk * (1.0 - np.cos(t13_c - k * np.pi))

    axes[2, 1].plot(t13_c, y13_c, 'b-', lw=1.5)
    axes[2, 1].axvline(15 * np.pi, color='r', linestyle='--', alpha=0.6, label="t = 15π")
    axes[2, 1].set_title("6.4 #13(c): Solution y(t) for n = 15 on [0, 60]")
    axes[2, 1].set_xlabel("t")
    axes[2, 1].set_ylabel("y(t)")
    axes[2, 1].legend()
    axes[2, 1].grid(True, linestyle=':')

    fig.subplots_adjust(hspace=0.4, wspace=0.25, top=0.92)
    pdf.savefig(fig)
    plt.close()

    # =========================================================================
    # PAGE 2: Section 6.5 (Problems 1, 4, 9, 13)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("Section 6.5: Impulse Functions", fontsize=14, fontweight='bold')

    # --- Problem 1 ---
    t_65_1 = np.linspace(0, 3 * np.pi, 1000)
    u_pi_1 = np.where(t_65_1 >= np.pi, 1.0, 0.0)
    y_65_1 = (np.exp(-t_65_1) * (np.cos(t_65_1) + np.sin(t_65_1)) 
              + u_pi_1 * np.exp(-np.maximum(0, t_65_1 - np.pi)) * np.sin(np.maximum(0, t_65_1 - np.pi)))
    axes[0, 0].plot(t_65_1, y_65_1, 'b-', lw=2)
    axes[0, 0].axvline(np.pi, color='r', linestyle='--', alpha=0.5, label="Impulse at t = π")
    axes[0, 0].set_title("6.5 #1(b): y'' + 2y' + 2y = δ(t - π)")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("y(t)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=':')

    # --- Problem 4 ---
    t_65_4 = np.linspace(0, 6 * np.pi, 1000)
    u_3pi_4 = np.where(t_65_4 >= 3 * np.pi, 1.0, 0.0)
    tau_4 = np.maximum(0, t_65_4 - 3 * np.pi)
    y_65_4 = (0.25 * (np.sin(t_65_4) - np.cos(t_65_4) + np.exp(-t_65_4) * np.cos(np.sqrt(2) * t_65_4))
              + (1.0 / np.sqrt(2)) * u_3pi_4 * np.exp(-tau_4) * np.sin(np.sqrt(2) * tau_4))
    axes[0, 1].plot(t_65_4, y_65_4, 'b-', lw=2)
    axes[0, 1].axvline(3 * np.pi, color='r', linestyle='--', alpha=0.5, label="Impulse at t = 3π")
    axes[0, 1].set_title("6.5 #4(b): y'' + 2y' + 3y = sin(t) + δ(t - 3π)")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("y(t)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':')

    # --- Problem 9 ---
    t_65_9 = np.linspace(0, 18, 1000)
    w = np.sqrt(15) / 4.0
    t0 = 5.0 + 8.0 * np.pi / np.sqrt(15)
    k_val = -np.exp(-2.0 * np.pi / np.sqrt(15))
    
    u5 = np.where(t_65_9 >= 5.0, 1.0, 0.0)
    ut0 = np.where(t_65_9 >= t0, 1.0, 0.0)
    tau5 = np.maximum(0, t_65_9 - 5.0)
    taut0 = np.maximum(0, t_65_9 - t0)
    
    y_65_9 = (2.0 / np.sqrt(15)) * (u5 * np.exp(-tau5 / 4.0) * np.sin(w * tau5)
                                    + k_val * ut0 * np.exp(-taut0 / 4.0) * np.sin(w * taut0))
    axes[1, 0].plot(t_65_9, y_65_9, 'b-', lw=2)
    axes[1, 0].axvline(5.0, color='g', linestyle='--', alpha=0.6, label="t = 5")
    axes[1, 0].axvline(t0, color='r', linestyle='--', alpha=0.6, label=f"t0 ≈ {t0:.2f}")
    axes[1, 0].set_title("6.5 #9(b): Oscillation Canceled at t0")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("y(t)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=':')

    # --- Problem 13 ---
    t_65_13 = np.linspace(0, 22 * np.pi, 2000)
    y_65_13 = np.zeros_like(t_65_13)
    for k in range(1, 21):
        uk = np.where(t_65_13 >= k * np.pi, 1.0, 0.0)
        y_65_13 += uk * np.sin(t_65_13 - k * np.pi)
    axes[1, 1].plot(t_65_13, y_65_13, 'b-', lw=1.5)
    axes[1, 1].set_title("6.5 #13(b): 20 Impulses at kπ")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("y(t)")
    axes[1, 1].grid(True, linestyle=':')

    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
    pdf.savefig(fig)
    plt.close()

    # =========================================================================
    # PAGE 3: Sections 7.5 & 7.6 (Problems 7.5 #1, 4 and 7.6 #1, 3)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Sections 7.5 & 7.6: Direction Fields & Trajectories", fontsize=14, fontweight='bold')

    plot_phase_portrait(axes[0, 0], np.array([[3, -2], [2, -2]]),
                        "7.5 #1: Saddle Point (r = 2, -1)",
                        eigvecs=[([2, 1], "ξ(1) (r=2)"), ([1, 2], "ξ(2) (r=-1)")])

    plot_phase_portrait(axes[0, 1], np.array([[1.25, 0.75], [0.75, 1.25]]),
                        "7.5 #4: Unstable Node (r = 2, 1/2)",
                        eigvecs=[([1, 1], "ξ(1) (r=2)"), ([1, -1], "ξ(2) (r=1/2)")])

    plot_phase_portrait(axes[1, 0], np.array([[-1, -4], [1, -1]]),
                        "7.6 #1: Spiral Sink (r = -1 ± 2i)")

    plot_phase_portrait(axes[1, 1], np.array([[1, -1], [5, -3]]),
                        "7.6 #3: Spiral Sink (r = -1 ± i)")

    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
    pdf.savefig(fig)
    plt.close()

    # =========================================================================
    # PAGE 4: Section 7.6 Problem 12(c) Bifurcation Phase Portraits
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Section 7.6 #12(c): Phase Portraits Across Bifurcations (±2√5 ≈ ±4.47, 0)",
                 fontsize=13, fontweight='bold')

    plot_phase_portrait(axes[0, 0], np.array([[0, -5], [1, -5]]),
                        "α = -5 (< -2√5): Stable Node (Sink)")

    plot_phase_portrait(axes[0, 1], np.array([[0, -5], [1, -2]]),
                        "α = -2 (-2√5 < α < 0): Spiral Sink")

    plot_phase_portrait(axes[1, 0], np.array([[0, -5], [1, 2]]),
                        "α = 2 (0 < α < 2√5): Spiral Source")

    plot_phase_portrait(axes[1, 1], np.array([[0, -5], [1, 5]]),
                        "α = 5 (> 2√5): Unstable Node (Source)")

    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
    pdf.savefig(fig)
    plt.close()

print(f"Successfully generated {pdf_filename}")