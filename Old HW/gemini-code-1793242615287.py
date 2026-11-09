import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HELPER FUNCTION TO GENERATE SMOOTH PV LOOPS
# =============================================================================
def make_pv_loop(esv, edv, p_edp, p_ao_open, p_peak, p_ao_close, p_min=5.0):
    """Generates (V, P) coordinates for a 4-phase cardiac PV loop."""
    # Phase I: Ventricular Filling (from ESV to EDV along diastolic curve)
    v_fill = np.linspace(esv, edv, 50)
    p_fill = p_min + (p_edp - p_min) * ((v_fill - esv) / (edv - esv))**2

    # Phase II: Isovolumic Contraction (vertical line up at EDV)
    v_iso_c = np.full(30, edv)
    p_iso_c = np.linspace(p_edp, p_ao_open, 30)

    # Phase III: Ventricular Ejection (arch from EDV to ESV)
    v_ejec = np.linspace(edv, esv, 60)
    #Normalized progress u from 0 (at EDV) to 1 (at ESV)
    u = (edv - v_ejec) / (edv - esv)
    p_ejec = (p_ao_open * (1 - u) + p_ao_close * u) + (p_peak - 0.5*(p_ao_open + p_ao_close)) * np.sin(np.pi * u)

    # Phase IV: Isovolumic Relaxation (vertical line down at ESV)
    v_iso_r = np.full(30, esv)
    p_iso_r = np.linspace(p_ao_close, p_min, 30)

    V = np.concatenate([v_fill, v_iso_c, v_ejec, v_iso_r])
    P = np.concatenate([p_fill, p_iso_c, p_ejec, p_iso_r])
    return V, P

# Passive Diastolic & Active Systolic Bounding Curves (from BME 244L Slides)
v_bound = np.linspace(0, 250, 200)
p_diastolic = 2.0 * np.exp(0.0185 * v_bound) - 2.0
p_systolic = 280.0 * np.sin(np.pi * (v_bound / 310.0))**1.6

# =============================================================================
# 1. PROBLEM 3: PRELOAD AND AFTERLOAD PV LOOPS (i, ii, iii)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Baseline (Normal) PV Loop
V_norm, P_norm = make_pv_loop(esv=50, edv=120, p_edp=8, p_ao_open=80, p_peak=120, p_ao_close=100)

# (i) Increased Preload (Higher EDV = 160 mL -> wider loop, larger SV)
V_pre, P_pre = make_pv_loop(esv=50, edv=160, p_edp=15, p_ao_open=80, p_peak=128, p_ao_close=100)

# (ii) Small Increase in Afterload (Higher aortic opening pressure & higher ESV = 70 mL)
V_aft1, P_aft1 = make_pv_loop(esv=70, edv=120, p_edp=8, p_ao_open=110, p_peak=150, p_ao_close=135)

# (iii) Extreme Increase in Afterload (Very high pressure & much higher ESV = 100 mL)
V_aft2, P_aft2 = make_pv_loop(esv=100, edv=120, p_edp=8, p_ao_open=180, p_peak=225, p_ao_close=210)

cases = [
    ("(i) Increased Preload", V_pre, P_pre, "red"),
    ("(ii) Small Increase in Afterload", V_aft1, P_aft1, "darkorange"),
    ("(iii) Extreme Increase in Afterload", V_aft2, P_aft2, "purple")
]

for ax, (title, V_mod, P_mod, col) in zip(axes, cases):
    ax.plot(v_bound, p_diastolic, 'b-', alpha=0.35, label="Diastolic Curve")
    ax.plot(v_bound, p_systolic, 'g-', alpha=0.35, label="Systolic Curve")
    ax.plot(V_norm, P_norm, 'k-', linewidth=2, label="Normal Loop")
    ax.plot(V_mod, P_mod, linestyle='--', color=col, linewidth=2.5, label=title)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Left Ventricular Volume (mL)")
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 300)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper left", fontsize=8)

axes[0].set_ylabel("Left Intraventricular Pressure (mm Hg)")
plt.suptitle("Problem 3: Effects of Preload and Afterload on PV Loops", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("Problem3_PV_Loops.png", dpi=150)

# =============================================================================
# 2. PROBLEM 7 (PARTS A & D): SPECIFIC PATIENT PV LOOPS
# =============================================================================
plt.figure(figsize=(7, 6))

# Optional bounding curves from lecture slides for visual reference
plt.plot(v_bound, p_diastolic, 'b-', alpha=0.25, label="Passive Diastolic Curve (Ref)")
plt.plot(v_bound, p_systolic, 'g-', alpha=0.25, label="Active Systolic Curve (Ref)")

# Part A: ESV = 60 mL, EDV = 125 mL, AV opens = 90 mmHg, AV closes = 125 mmHg
V_7a, P_7a = make_pv_loop(esv=60, edv=125, p_edp=8, p_ao_open=90, p_peak=138, p_ao_close=125)
plt.plot(V_7a, P_7a, 'k-', linewidth=2.5, label="Part A: Resting (EDV=125, ESV=60, SV=65 mL)")

# Part D: EDV increases to 150 mL (dotted line)
V_7d, P_7d = make_pv_loop(esv=60, edv=150, p_edp=12, p_ao_open=90, p_peak=145, p_ao_close=125)
plt.plot(V_7d, P_7d, 'r:', linewidth=3.0, label="Part D: Increased Preload (EDV=150, ESV=60, SV=90 mL)")

# Annotate key valve points
plt.plot([125, 125, 60, 60], [8, 90, 125, 5], 'ko', markersize=6)
plt.annotate("AV Opens (125 mL, 90 mmHg)", xy=(125, 90), xytext=(140, 85),
             arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
plt.annotate("AV Closes (60 mL, 125 mmHg)", xy=(60, 125), xytext=(10, 155),
             arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
plt.annotate("New EDV (150 mL)", xy=(150, 12), xytext=(165, 25),
             arrowprops=dict(arrowstyle="->", color='red', lw=1), color='red', fontsize=9)

plt.title("Problem 7 (Parts A & D): Left Ventricular PV Loops", fontweight='bold')
plt.xlabel("Left ventricular volume (ml)")
plt.ylabel("Left intraventricular pressure (mm Hg)")
plt.xlim(0, 250)
plt.ylim(0, 300)
plt.xticks(np.arange(0, 251, 50))
plt.yticks(np.arange(0, 301, 50))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("Problem7_PV_Loop.png", dpi=150)

# =============================================================================
# 3. PROBLEM 8 (PART a): RIGHT HEART PRESSURES & SA NODE ARREST
# =============================================================================
t = np.linspace(0, 1.0, 500)

# Original Right Ventricle (RV) pressure profile (systole from ~0.10s to 0.42s)
P_rv_orig = np.where((t >= 0.10) & (t <= 0.42),
                     3.0 + 24.0 * np.sin(np.pi * (t - 0.10) / 0.32),
                     3.0 + 1.0 * np.sin(2 * np.pi * t))

# Original Pulmonary Artery (PA) pressure profile
P_pa = np.where((t >= 0.16) & (t <= 0.40),
                27.0 * np.sin(np.pi * (t - 0.10) / 0.32),
                15.0 * np.exp(-1.2 * ((t - 0.40) % 1.0)))
P_pa = np.maximum(P_pa, 10.0)

# Original Right Atrium (RA) pressure profile
P_ra = 4.0 + 2.0 * np.exp(-((t - 0.08)/0.04)**2) + 1.5 * np.exp(-((t - 0.38)/0.05)**2)

# Sketched Option 1: Delayed AV Node Escape Beat (shifted right, missing a-wave kick)
P_rv_av_escape = np.where((t >= 0.55) & (t <= 0.87),
                          3.0 + 22.0 * np.sin(np.pi * (t - 0.55) / 0.32),
                          3.0)

# Sketched Option 2: Flatline prior to escape beat
P_rv_flat = np.full_like(t, 3.0)

plt.figure(figsize=(8, 5))
plt.plot(t, P_pa, 'b--', linewidth=1.8, alpha=0.7, label="Pulmonary Artery (Original)")
plt.plot(t, P_rv_orig, 'k-', linewidth=2.2, label="Right Ventricle (Original)")
plt.plot(t, P_ra, 'g-', linewidth=1.8, alpha=0.8, label="Right Atrium (Original)")

# Plot the two valid ways to sketch 8a
plt.plot(t, P_rv_av_escape, 'r--', linewidth=2.5, label="Sketch Option 1: Delayed RV Beat (AV Node Takeover)")
plt.plot(t, P_rv_flat, 'm:', linewidth=2.5, label="Sketch Option 2: Flat RV Baseline (Before Escape Beat)")

# Mark Tricuspid/Mitral Valve Opening at t ~ 0.42 s (Problem 8c)
plt.plot(0.415, 4.5, 'ro', markersize=7)
plt.annotate("AV (Tricuspid/Mitral) Valve Opens\n(t ≈ 0.42 s)", xy=(0.415, 4.5), xytext=(0.46, 12),
             arrowprops=dict(arrowstyle="->", color='black'), fontsize=9)

plt.title("Problem 8a & 8c: Right Heart Pressure Profiles", fontweight='bold')
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mm Hg)")
plt.xlim(0, 1.0)
plt.ylim(0, 40)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="upper right", fontsize=8.5)
plt.tight_layout()
plt.savefig("Problem8_Pressures.png", dpi=150)

plt.show()