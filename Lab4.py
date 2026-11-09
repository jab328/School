# ==============================================================================
# AI Attribution:
# Gemini helped draft this code.
# All code was reviewed, tested, and verified by Junior Bonet.
# ==============================================================================

"""
EGR 201L - Lab 4: Torsion Test Data Analysis and Plotting
Author: Junior Bonet (jab328@duke.edu)
Lab Partners: Kevin Nguyen (kn213@duke.edu), Ryan Davis (rcd50@duke.edu)
Date: November 8, 2026
"""

import numpy as np
import matplotlib.pyplot as plt


def load_torsion_data(filepath):
    """
    Loads raw Angle of Twist (deg) and Torque (N-m) data from the Horizon export
    and filters out post-fracture negative torque readings.
    """
    raw_data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    angle_deg = raw_data[:, 0]
    torque_nm = raw_data[:, 1]

    # Retain data from test start up to fracture (before torque drops negative)
    valid_mask = torque_nm >= 0.0
    return angle_deg[valid_mask], torque_nm[valid_mask]


def analyze_torsion_properties(angle_deg, torque_nm, d_mm=19.61, L_mm=183.0):
    """
    Computes shear strain, shear stress, linear slope (T/gamma), shear modulus (G),
    0.2% offset yield torque/stress, and ultimate torque/stress.
    """
    # Convert geometric dimensions from mm to meters
    r_m = (d_mm / 2.0) * 1e-3
    L_m = L_mm * 1e-3

    # Polar moment of inertia J (m^4) for a solid circular shaft
    J_m4 = (np.pi * (r_m ** 4)) / 2.0

    # 1. Convert rotation readings phi from degrees to radians
    phi_rad = np.deg2rad(angle_deg)

    # 2. Calculate surface shear strain gamma = (r * phi) / L
    gamma = (r_m * phi_rad) / L_m

    # Calculate surface shear stress tau = (T * r) / J in MPa
    tau_mpa = (torque_nm * r_m / J_m4) * 1e-6

    # Estimate initial pre-slip slope (0 <= phi <= 0.431 deg)
    init_mask = (angle_deg >= 0.04) & (angle_deg <= 0.431)
    slope_init, _ = np.polyfit(gamma[init_mask], torque_nm[init_mask], 1)
    G_init_gpa = (slope_init * r_m / J_m4) * 1e-9

    # 5. Estimate ratio T/gamma on the steady post-slip linear region
    # (2.60 deg <= phi <= 7.60 deg, i.e., 0.00243 <= gamma <= 0.00711)
    linear_mask = (gamma >= 0.00243) & (gamma <= 0.00711)
    slope_T_gamma, intercept_T = np.polyfit(
        gamma[linear_mask], torque_nm[linear_mask], 1
    )

    # Calculate experimental Shear Modulus G (in GPa)
    G_gpa = (slope_T_gamma * r_m / J_m4) * 1e-9

    # Construct 0.2% (0.002 rad) offset line: T_offset = (T/gamma) * (gamma - 0.002)
    offset_strain = 0.002
    T_offset_line = slope_T_gamma * (gamma - offset_strain)

    # Locate intersection of 0.2% offset line with experimental curve (gamma > 0.007)
    search_mask = gamma >= 0.007
    search_indices = np.where(search_mask)[0]
    diff = torque_nm[search_indices] - T_offset_line[search_indices]

    # Find index where experimental curve crosses below the offset line
    cross_idx = search_indices[np.where(diff <= 0)[0][0]]
    i1, i2 = cross_idx - 1, cross_idx

    # Linearly interpolate exact yield strain (gamma_y) and yield torque (T_y)
    frac = diff[i1 - search_indices[0]] / (
        diff[i1 - search_indices[0]] - diff[i2 - search_indices[0]]
    )
    gamma_y = gamma[i1] + frac * (gamma[i2] - gamma[i1])
    T_y = torque_nm[i1] + frac * (torque_nm[i2] - torque_nm[i1])
    tau_y_mpa = (T_y * r_m / J_m4) * 1e-6

    # Ultimate torque (T_u) and ultimate shear stress (tau_u)
    ult_idx = np.argmax(torque_nm)
    gamma_u = gamma[ult_idx]
    T_u = torque_nm[ult_idx]
    tau_u_elastic_mpa = (T_u * r_m / J_m4) * 1e-6
    tau_u_plastic_mpa = 0.75 * tau_u_elastic_mpa

    results = {
        "r_m": r_m,
        "L_m": L_m,
        "J_m4": J_m4,
        "phi_rad": phi_rad,
        "gamma": gamma,
        "tau_mpa": tau_mpa,
        "slope_init": slope_init,
        "G_init_gpa": G_init_gpa,
        "slope_T_gamma": slope_T_gamma,
        "intercept_T": intercept_T,
        "G_gpa": G_gpa,
        "gamma_y": gamma_y,
        "T_y": T_y,
        "tau_y_mpa": tau_y_mpa,
        "gamma_u": gamma_u,
        "T_u": T_u,
        "tau_u_elastic_mpa": tau_u_elastic_mpa,
        "tau_u_plastic_mpa": tau_u_plastic_mpa,
    }
    return results


def plot_torsion_curves(torque_nm, results):
    """
    Generates separate plots for:
    (a) Full range of shear strain (from 0 to fracture)
    (b) Reduced range of shear strain (from 0 to 0.013, just past yield strain ~0.012)
    """
    gamma = results["gamma"]
    slope = results["slope_T_gamma"]
    gamma_y = results["gamma_y"]
    T_y = results["T_y"]
    gamma_u = results["gamma_u"]
    T_u = results["T_u"]
    r_m = results["r_m"]
    J_m4 = results["J_m4"]

    # --------------------------------------------------------------------------
    # Plot 1: Full Range Torque & Shear Stress vs. Shear Strain
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    ax1.plot(
        gamma, torque_nm, color="#1f77b4", linewidth=2.0, label="Experimental Data"
    )
    ax1.plot(
        gamma_y,
        T_y,
        "ro",
        markersize=7,
        label=f"0.2% Offset Yield ($\\gamma_y={gamma_y:.4f}$, $T_y={T_y:.1f}$ N-m)",
    )
    ax1.plot(
        gamma_u,
        T_u,
        "kx",
        markersize=8,
        markeredgewidth=2.0,
        label=f"Ultimate Torque ($\\gamma_u={gamma_u:.4f}$, $T_u={T_u:.1f}$ N-m)",
    )

    ax1.set_xlabel("Shear Strain, $\\gamma$ (rad)", fontsize=11)
    ax1.set_ylabel("Torque, $T$ (N-m)", fontsize=11)
    ax1.set_title(
        "Figure 1: Full-Range Torque and Shear Stress vs. Shear Strain (6061-T6 Al)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlim(0, 0.155)
    ax1.set_ylim(0, 410)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Secondary y-axis showing elastic-formula Shear Stress tau (MPa)
    ax1_stress = ax1.twinx()
    ax1_stress.set_ylim(0, 410 * (r_m / J_m4) * 1e-6)
    ax1_stress.set_ylabel("Shear Stress, $\\tau = Tr/J$ (MPa)", fontsize=11)

    ax1.legend(loc="lower right", frameon=True)
    fig1.tight_layout()
    fig1.savefig("torsion_full_range.png", dpi=300)

    # --------------------------------------------------------------------------
    # Plot 2: Reduced Range Torque & Shear Stress vs. Shear Strain (0 to 0.013)
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    red_mask = gamma <= 0.0135

    ax2.plot(
        gamma[red_mask],
        torque_nm[red_mask],
        color="#1f77b4",
        linewidth=2.0,
        label="Experimental Data",
    )

    # Linear elastic slope reference line
    gamma_lin = np.linspace(0, 0.0115, 100)
    ax2.plot(
        gamma_lin,
        slope * gamma_lin,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label=f"Linear Fit ($T/\\gamma = {slope:,.0f}$ N-m/rad, $G = {results['G_gpa']:.2f}$ GPa)",
    )

    # 0.2% strain offset line
    gamma_off = np.linspace(0.002, 0.0130, 100)
    ax2.plot(
        gamma_off,
        slope * (gamma_off - 0.002),
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
        label="0.2% Strain Offset Line ($\\gamma_{offset} = 0.002$)",
    )

    # Highlight Yield Point and Slip Event 2
    ax2.plot(
        gamma_y,
        T_y,
        "ro",
        markersize=7,
        label=f"Yield Point ($\\gamma_y={gamma_y:.4f}$, $T_y={T_y:.1f}$ N-m, $\\tau_y={results['tau_y_mpa']:.1f}$ MPa)",
    )
    ax2.annotate(
        "Grip Slip 2\n($\\phi=2.36^\\circ, T=68.5$ N-m)",
        xy=(0.00221, 68.47),
        xytext=(0.0035, 45.0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        fontsize=9,
    )

    ax2.set_xlabel("Shear Strain, $\\gamma$ (rad)", fontsize=11)
    ax2.set_ylabel("Torque, $T$ (N-m)", fontsize=11)
    ax2.set_title(
        "Figure 2: Reduced-Range Torque and Shear Stress vs. Shear Strain (6061-T6 Al)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlim(0, 0.0135)
    ax2.set_ylim(0, 340)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Secondary y-axis showing Shear Stress tau (MPa)
    ax2_stress = ax2.twinx()
    ax2_stress.set_ylim(0, 340 * (r_m / J_m4) * 1e-6)
    ax2_stress.set_ylabel("Shear Stress, $\\tau = Tr/J$ (MPa)", fontsize=11)

    ax2.legend(loc="lower right", fontsize=9, frameon=True)
    fig2.tight_layout()
    fig2.savefig("torsion_reduced_range.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    data_file = "Torsion Data_kn213_A_6061 T6 Aluminum.txt"
    angle_deg, torque_nm = load_torsion_data(data_file)
    res = analyze_torsion_properties(angle_deg, torque_nm, d_mm=19.61, L_mm=183.0)

    print("=== EGR 201L Lab 4 Torsion Analysis Summary (6061-T6 Aluminum) ===")
    print(f"Polar Moment of Inertia (J):       {res['J_m4']:.4e} m^4")
    print(f"Initial Pre-Slip Slope (T/gamma):  {res['slope_init']:,.1f} N-m/rad")
    print(f"Initial Pre-Slip Shear Modulus G:  {res['G_init_gpa']:.2f} GPa")
    print(f"Post-Slip Linear Slope (T/gamma):  {res['slope_T_gamma']:,.1f} N-m/rad")
    print(f"Experimental Shear Modulus (G):    {res['G_gpa']:.2f} GPa")
    print(f"0.2% Offset Yield Strain (gamma_y):{res['gamma_y']:.5f} rad")
    print(f"0.2% Offset Yield Torque (T_y):    {res['T_y']:.2f} N-m")
    print(f"0.2% Offset Yield Stress (tau_y):  {res['tau_y_mpa']:.2f} MPa")
    print(f"Ultimate Torque (T_u):             {res['T_u']:.2f} N-m")
    print(f"Ultimate Shear Stress (Elastic):   {res['tau_u_elastic_mpa']:.2f} MPa")
    print(f"Ultimate Shear Stress (Plastic):   {res['tau_u_plastic_mpa']:.2f} MPa")

    plot_torsion_curves(torque_nm, res)