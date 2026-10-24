import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. LOCATE FOLDER & LOAD Exp2_Force_Velocity_Results.csv
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'Exp2_Force_Velocity_Results.csv')

# Fitted Hill parameters from Question 4 & Isometric Muscle Length from Appendix A
F0 = 2.4611e6       # Estimated isometric force (dynes)
a_param = 4.2435e5  # Hill constant a (dynes)
b_param = 2.4148    # Hill constant b (cm/s)
L_iso_cm = 4.842    # Isometric muscle length (48.42 mm = 4.842 cm)

# Experiment 1 Data (Appendix A)
exp1_volts = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
exp1_disp_mm = np.array([0.0, 0.0, 2.86, 4.02, 4.09, 4.23, 4.63, 4.85, 4.91, 4.93, 5.07, 4.99])

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Make sure Exp2_Force_Velocity_Results.csv is in {SCRIPT_DIR}")

# Load CSV and remove duplicate rows caused by case-insensitive globbing
df_csv = pd.read_csv(CSV_PATH)
df_csv['Trial_ID'] = df_csv['Trial_ID'].astype(str).str.zfill(6)
df_csv = df_csv.drop_duplicates(subset=['Trial_ID']).copy()

# Filter out crossed-out/superseded repeat trials for Question 7 (25 valid loads)
invalid_ids = ['182929', '183024', '183418', '183514', '184202']
df_valid = df_csv[~df_csv['Trial_ID'].isin(invalid_ids)].sort_values('Load_g').copy()

# Filter the 8 evenly-spaced trials used in Question 2 & Question 3 Hill Fit
q2_ids = ['182007', '182608', '182758', '183134', '183741', '184023', '184242', '184557']
df_q2 = df_csv[df_csv['Trial_ID'].isin(q2_ids)].sort_values('Load_g').copy()

print(f"\nSuccessfully loaded {len(df_csv)} unique trials from: {CSV_PATH}")
print("\n=== QUESTION 2 TABLE (EXACT VALUES FROM YOUR CSV) ===")
print(df_q2[['Trial_ID', 'Load_g', 'Force_dynes', 'Formatted_Velocity', 'R_squared']].to_string(index=False))

# =====================================================================
# 2. EXPERIMENT 1, QUESTION 1: STIMULUS-RESPONSE GRAPH
# =====================================================================
plt.figure(figsize=(7, 4.5))
plt.plot(exp1_volts, exp1_disp_mm, 'bo-', linewidth=1.5, markersize=6, label='Experimental Data')
plt.xlabel('Stimulus Strength (V)', fontsize=12)
plt.ylabel('Muscle Displacement (mm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
out_q1 = os.path.join(SCRIPT_DIR, 'Exp1_Q1_Stimulus_Response.png')
plt.savefig(out_q1, dpi=300)
plt.close()

# =====================================================================
# 3. EXPERIMENT 2, QUESTION 7: EXCESS HEAT VS. DISPLACEMENT GRAPH
# =====================================================================
disp_cm_all = df_valid['Reported_Disp_mm'].values / 10.0
excess_heat_all = a_param * disp_cm_all  # Q = a * delta_x (ergs)

plt.figure(figsize=(7, 4.5))
plt.plot(disp_cm_all, excess_heat_all, 'ko-', linewidth=1.5, markersize=5,
         label=r'Estimated Excess Heat ($Q = a \cdot \Delta x$)')
plt.xlabel('Muscle Displacement (cm)', fontsize=12)
plt.ylabel('Excess Heat Released (ergs)', fontsize=12)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
out_q7 = os.path.join(SCRIPT_DIR, 'Exp2_Q7_Excess_Heat_vs_Displacement.png')
plt.savefig(out_q7, dpi=300)
plt.close()

# =====================================================================
# 4. EXPERIMENT 2, QUESTION 9: MUSCLE POWER VS. VELOCITY GRAPH
# =====================================================================
q2_forces = df_q2['Force_dynes'].values
q2_velocities = df_q2['Velocity_cms'].values
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
out_q9 = os.path.join(SCRIPT_DIR, 'Exp2_Q9_Power_vs_Velocity.png')
plt.savefig(out_q9, dpi=300)
plt.close()

# =====================================================================
# 5. EXPERIMENT 2, QUESTION 11: ISOMETRIC FORCE VS. TIME GRAPH
# =====================================================================
# From BME 244L lecture slide: C = 0.1 * Isometric_Muscle_Length, alpha = F0 / C
C = 0.1 * L_iso_cm
alpha = F0 / C

F_vals = np.linspace(0, 0.995 * F0, 500)
t_vals = (1.0 / (alpha * b_param)) * ((F0 + a_param) * np.log(F0 / (F0 - F_vals)) - F_vals)

plt.figure(figsize=(7, 4.5))
plt.plot(t_vals, F_vals, 'g-', linewidth=2, label="Hill's 2-Component Isometric Prediction")
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Force (dynes)', fontsize=12)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
out_q11 = os.path.join(SCRIPT_DIR, 'Exp2_Q11_Isometric_Force_vs_Time.png')
plt.savefig(out_q11, dpi=300)
plt.close()

print(f"\nSaved all 4 plots directly to: {SCRIPT_DIR}")