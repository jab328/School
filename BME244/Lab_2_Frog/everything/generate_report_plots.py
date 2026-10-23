import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. PARAMETERS & EXPERIMENTAL DATA (From Appendix A & Question 4)
# =====================================================================
F0 = 2.4611e6       # Estimated isometric force (dynes)
a_param = 4.2435e5  # Hill constant a (dynes)
b_param = 2.4148    # Hill constant b (cm/s)
L_iso_cm = 4.842    # Isometric muscle length (48.42 mm = 4.842 cm)
G_CGS = 980.0       # cm/s^2

# Experiment 1 Data (Appendix A)
exp1_volts = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
exp1_disp_mm = np.array([0.0, 0.0, 2.86, 4.02, 4.09, 4.23, 4.63, 4.85, 4.91, 4.93, 5.07, 4.99])

# Experiment 2 Valid Trials (Appendix A: Load in g, Displacement in mm)
exp2_valid = [
    ('182007', 0, 4.66), ('182135', 50, 4.55), ('182224', 100, 4.31),
    ('182416', 150, 3.99), ('182452', 200, 3.71), ('182608', 300, 3.03),
    ('182646', 400, 2.53), ('182714', 500, 2.51), ('182758', 600, 2.40),
    ('182842', 700, 2.45), ('183242', 800, 1.82), ('183134', 900, 0.99),
    ('183321', 1000, 0.73), ('183651', 1100, 1.16), ('183741', 1200, 1.18),
    ('183848', 1300, 0.68), ('183933', 1400, 0.71), ('184023', 1500, 0.62),
    ('184054', 1600, 0.37), ('184132', 1700, 0.19), ('184242', 1800, 0.44),
    ('184325', 1900, 0.31), ('184400', 2000, 0.19), ('184511', 2100, 0.08),
    ('184557', 2200, 0.02)
]

# The 8 trials used in Question 2 & Question 3 Hill fit
q2_ids = ['182007', '182608', '182758', '183134', '183741', '184023', '184242', '184557']
q2_loads_g = np.array([0, 300, 600, 900, 1200, 1500, 1800, 2200], dtype=float)
q2_forces = q2_loads_g * G_CGS
q2_velocities = np.array([14.07, 6.86, 5.18, 2.42, 2.17, 1.31, 0.92, 0.00])

# Load exact velocities & 95% CLs from Exp2_Force_Velocity_Results.csv if present
csv_path = 'Exp2_Force_Velocity_Results.csv'
if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    df_csv['Trial_ID'] = df_csv['Trial_ID'].astype(str).str.zfill(6)
    sub = df_csv[df_csv['Trial_ID'].isin(q2_ids)].copy()
    if len(sub) == 8:
        q2_forces = sub['Force_dynes'].values
        q2_velocities = sub['Velocity_cms'].values
        print("\n=== QUESTION 2 TABLE (FROM YOUR CSV) ===")
        print(sub[['Trial_ID', 'Load_g', 'Force_dynes', 'Formatted_Velocity', 'R_squared']].to_string(index=False))

# =====================================================================
# 2. EXPERIMENT 1, QUESTION 1: STIMULUS-RESPONSE GRAPH
# =====================================================================
plt.figure(figsize=(7, 4.5))
plt.plot(exp1_volts, exp1_disp_mm, 'bo-', linewidth=1.5, markersize=6, label='Experimental Data')
plt.xlabel('Stimulus Strength (V)', fontsize=12)
plt.ylabel('Muscle Displacement (mm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Exp1_Q1_Stimulus_Response.png', dpi=300)
plt.close()

# =====================================================================
# 3. EXPERIMENT 2, QUESTION 7: EXCESS HEAT VS. DISPLACEMENT GRAPH
# =====================================================================
disp_cm_all = np.array([row[2] / 10.0 for row in exp2_valid])
excess_heat_all = a_param * disp_cm_all  # Q = a * delta_x (ergs)

plt.figure(figsize=(7, 4.5))
plt.plot(disp_cm_all, excess_heat_all, 'ko-', linewidth=1.5, markersize=5, label=r'Estimated Excess Heat ($Q = a \cdot \Delta x$)')
plt.xlabel('Muscle Displacement (cm)', fontsize=12)
plt.ylabel('Excess Heat Released (ergs)', fontsize=12)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('Exp2_Q7_Excess_Heat_vs_Displacement.png', dpi=300)
plt.close()

# =====================================================================
# 4. EXPERIMENT 2, QUESTION 9: MUSCLE POWER VS. VELOCITY GRAPH
# =====================================================================
exp_power = q2_forces * q2_velocities  # P = F * v (erg/s)

v_max_hill = (b_param * F0) / a_param
v_line = np.linspace(0, v_max_hill, 300)
hill_power = (v_line * (b_param * F0 - a_param * v_line)) / (v_line + b_param)

plt.figure(figsize=(7, 4.5))
plt.plot(q2_velocities, exp_power, 'ro', markersize=6, label='Experimental Data')
plt.plot(v_line, hill_power, 'b-', linewidth=2, label='Hill Model Prediction')
plt.xlabel('Contraction Velocity (cm/s)', fontsize=12)
plt.ylabel('Muscle Power (erg/s)', fontsize=12)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('Exp2_Q9_Power_vs_Velocity.png', dpi=300)
plt.close()

# =====================================================================
# 5. EXPERIMENT 2, QUESTION 11: ISOMETRIC FORCE VS. TIME GRAPH
# =====================================================================
# Series elastic stiffness k = F0 / delta_L_se_max
# Using total isometric muscle length L_iso_cm = 4.842 cm (adjust se_fraction if specified in lecture notes, e.g., 1.0 or 0.10)
se_fraction = 1.0
k_spring = F0 / (se_fraction * L_iso_cm)

F_vals = np.linspace(0, 0.995 * F0, 500)
t_vals = (1.0 / (k_spring * b_param)) * ((F0 + a_param) * np.log(F0 / (F0 - F_vals)) - F_vals)

plt.figure(figsize=(7, 4.5))
plt.plot(t_vals, F_vals, 'g-', linewidth=2, label="Hill's 2-Component Isometric Prediction")
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Force (dynes)', fontsize=12)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('Exp2_Q11_Isometric_Force_vs_Time.png', dpi=300)
plt.close()

print("Saved all plots: Exp1_Q1, Exp2_Q7, Exp2_Q9, and Exp2_Q11!")