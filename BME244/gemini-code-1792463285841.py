import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =====================================================================
# 1. YOUR EXPERIMENT 2 TRIAL METADATA (From Appendix A)
# =====================================================================
# Format: 'Experiment_ID': (Load_in_grams, Reported_Displacement_mm, 'Note')
EXP2_TRIALS = {
    '182007': (0, 4.66, '0g (No load)'),
    '182135': (50, 4.55, ''),
    '182224': (100, 4.31, ''),
    '182416': (150, 3.99, ''),
    '182452': (200, 3.71, ''),
    '182608': (300, 3.03, ''),
    '182646': (400, 2.53, ''),
    '182714': (500, 2.51, ''),
    '182758': (600, 2.40, ''),
    '182842': (700, 2.45, ''),
    '182929': (800, 1.96, ''),
    '183024': (900, 1.98, 'Rerun - data makes no sense'),
    '183134': (900, 0.99, '900g (repeat)'),
    '183242': (800, 1.82, '800g (repeat - right)'),
    '183321': (1000, 0.73, ''),
    '183418': (1100, 1.25, '1100g (run 1)'),
    '183514': (1100, 1.22, '1100g (repeat 2)'),
    '183651': (1100, 1.16, '1100g (repeat 3)'),
    '183741': (1200, 1.18, ''),
    '183848': (1300, 0.68, ''),
    '183933': (1400, 0.71, ''),
    '184023': (1500, 0.62, ''),
    '184054': (1600, 0.37, ''),
    '184132': (1700, 0.19, ''),
    '184202': (1800, 0.00, 'Crossed out'),
    '184242': (1800, 0.44, '1800g (repeat)'),
    '184325': (1900, 0.31, ''),
    '184400': (2000, 0.19, ''),
    '184511': (2100, 0.08, ''),
    '184557': (2200, 0.02, 'Isometric (~0 displacement)')
}

G_CGS = 980.0  # Acceleration due to gravity in cm/s^2 (1 g * 980 cm/s^2 = 980 dynes)

# =====================================================================
# 2. FUNCTION TO PARSE LABVIEW (.LVM) FILES
# =====================================================================
def parse_lvm(filepath):
    """
    Parses a LabVIEW Measurement (.lvm) file with two '***End_of_Header***' blocks
    and extracts Time (ms), Raw V, and Calculated mm.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find the line index after the last '***End_of_Header***'
    header_ends = [i for i, line in enumerate(lines) if '***End_of_Header***' in line]
    if not header_ends:
        raise ValueError(f"No LabVIEW header found in {filepath}")
    
    data_start_idx = header_ends[-1] + 2  # Skip '***End_of_Header***' and column names row

    time_ms = []
    raw_v = []
    calc_mm = []

    for line in lines[data_start_idx:]:
        if not line.strip():
            continue
        # Splitting by tab; leading tab creates an empty string at parts[0]
        parts = line.strip('\r\n').split('\t')
        # Filter out empty strings if leading tab is present, or index directly
        nums = [p.strip() for p in parts if p.strip() != '']
        if len(nums) >= 3:
            try:
                time_ms.append(float(nums[0]))
                raw_v.append(float(nums[1]))
                calc_mm.append(float(nums[2]))
            except ValueError:
                continue

    return pd.DataFrame({
        'Time_ms': time_ms,
        'Time_s': np.array(time_ms) / 1000.0,
        'Raw_V': raw_v,
        'Calc_mm': calc_mm,
        'Calc_cm': np.array(calc_mm) / 10.0
    })

# =====================================================================
# 3. FUNCTION TO FIND LINEAR RISING PHASE & 95% CONFIDENCE LIMITS
# =====================================================================
def analyze_contraction(df, reported_disp_mm=None):
    """
    Finds the most linear rising portion of positive muscle displacement,
    calculates velocity (cm/s), 95% confidence limits, and R^2.
    """
    t_s = df['Time_s'].values
    y_cm = df['Calc_cm'].values

    # Baseline is the average position over the first 50 ms
    baseline_cm = np.mean(y_cm[:50])
    disp_cm = y_cm - baseline_cm

    # If the transducer records shortening as a decrease in mm, flip sign so shortening is positive
    if np.max(np.abs(disp_cm)) == np.abs(np.min(disp_cm)) and np.max(disp_cm) < 0.02:
        disp_cm = -disp_cm

    max_disp_cm = np.max(disp_cm)
    peak_idx = np.argmax(disp_cm)

    # Check for isometric / negligible displacement (< 0.05 mm)
    if (reported_disp_mm is not None and reported_disp_mm <= 0.03) or max_disp_cm < 0.005:
        return {
            'velocity_cms': 0.0,
            'cl_95_cms': 0.0,
            'r_squared': np.nan,
            'n_points': 0,
            'slope': 0.0,
            'intercept': 0.0,
            't_fit': np.array([]),
            'y_fit': np.array([]),
            'disp_cm': disp_cm
        }

    # Isolate the rising phase between 20% and 80% of peak displacement prior to peak_idx
    rising_disp = disp_cm[:peak_idx + 1]
    low_thresh = 0.20 * max_disp_cm
    high_thresh = 0.80 * max_disp_cm

    # Last time it crosses 20% on the way up to first time it reaches 80%
    below_20 = np.where(rising_disp <= low_thresh)[0]
    start_idx = below_20[-1] if len(below_20) > 0 else 0

    above_80 = np.where(rising_disp >= high_thresh)[0]
    end_idx = above_80[0] if len(above_80) > 0 else peak_idx

    # Ensure at least 5 sample points (5 ms) for regression
    if end_idx - start_idx < 4:
        start_idx = max(0, peak_idx - 10)
        end_idx = peak_idx

    t_window = t_s[start_idx:end_idx + 1]
    y_window = disp_cm[start_idx:end_idx + 1]
    n = len(t_window)

    # Perform ordinary least squares linear regression
    res = stats.linregress(t_window, y_window)
    velocity_cms = res.slope
    se_slope = res.stderr
    r_squared = res.rvalue ** 2

    # 95% Confidence Limit: t_{0.05(2), df=n-2} * SE_slope
    t_crit = stats.t.ppf(0.975, df=n - 2)
    cl_95 = t_crit * se_slope

    return {
        'velocity_cms': velocity_cms,
        'cl_95_cms': cl_95,
        'se_slope': se_slope,
        't_crit': t_crit,
        'r_squared': r_squared,
        'n_points': n,
        'slope': res.slope,
        'intercept': res.intercept,
        't_fit': t_window,
        'y_fit': y_window,
        'disp_cm': disp_cm
    }

# =====================================================================
# 4. BATCH PROCESS FILES & GENERATE APPENDIX B PLOT
# =====================================================================
def main(data_folder='.'):
    lvm_files = sorted(glob.glob(os.path.join(data_folder, '*.lvm')) + 
                       glob.glob(os.path.join(data_folder, '*.LVM')))
    
    if not lvm_files:
        print(f"No .lvm files found in '{os.path.abspath(data_folder)}'.")
        print("Make sure this script is in the same folder as your .lvm files!")
        return

    results = []
    example_plotted = False

    for filepath in lvm_files:
        filename = os.path.basename(filepath)
        # Extract the 6-digit trial ID (e.g., '182007' from '182007.lvm')
        trial_id = ''.join(filter(str.isdigit, os.path.splitext(filename)[0]))[-6:]

        df = parse_lvm(filepath)

        if trial_id in EXP2_TRIALS:
            load_g, rep_disp_mm, note = EXP2_TRIALS[trial_id]
            force_dynes = load_g * G_CGS
            fit = analyze_contraction(df, reported_disp_mm=rep_disp_mm)

            results.append({
                'Trial_ID': trial_id,
                'Load_g': load_g,
                'Force_dynes': force_dynes,
                'Velocity_cms': fit['velocity_cms'],
                'CL_95_cms': fit['cl_95_cms'],
                'Formatted_Velocity': f"{fit['velocity_cms']:.2f} (± {fit['cl_95_cms']:.2f})",
                'R_squared': fit['r_squared'],
                'N_points': fit['n_points'],
                'Reported_Disp_mm': rep_disp_mm,
                'Notes': note
            })

            # Save an Appendix B example plot for the 0g or 300g trial
            if not example_plotted and load_g in (0, 300) and fit['n_points'] >= 5:
                t_s = df['Time_s'].values
                disp_cm = fit['disp_cm']
                t_fit = fit['t_fit']
                y_pred = fit['slope'] * t_fit + fit['intercept']

                plt.figure(figsize=(8, 5))
                # Zoom around the contraction event
                mid_t = np.mean(t_fit)
                mask = (t_s >= max(0, mid_t - 0.15)) & (t_s <= min(2.0, mid_t + 0.25))

                plt.plot(t_s[mask], disp_cm[mask], 'o', color='lightgray', markersize=4, label='Raw Displacement Data')
                plt.plot(t_fit, fit['y_fit'], 'bo', markersize=5, label='Linear Rising Portion (20%–80%)')
                
                eq_label = (f"Least Squares Fit: y = {fit['slope']:.3f}x + ({fit['intercept']:.3f})\n"
                            f"$R^2$ = {fit['r_squared']:.4f} | v = {fit['velocity_cms']:.2f} ± {fit['cl_95_cms']:.2f} cm/s")
                plt.plot(t_fit, y_pred, 'r-', linewidth=2, label=eq_label)

                plt.xlabel('Time (s)', fontsize=12)
                plt.ylabel('Muscle Displacement (cm)', fontsize=12)
                plt.legend(loc='best', fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(f'AppendixB_Velocity_Fit_{trial_id}.png', dpi=300)
                plt.close()
                example_plotted = True
                print(f"Saved Appendix B example graph: AppendixB_Velocity_Fit_{trial_id}.png")
                print(f"--- Appendix B Sample Math (Trial {trial_id}, Load = {load_g} g) ---")
                print(f"  Force = {load_g} g * 980 cm/s^2 = {force_dynes:.0f} dynes")
                print(f"  Slope (Velocity) = {fit['velocity_cms']:.4f} cm/s")
                print(f"  SE of Slope = {fit.get('se_slope', 0):.4f} cm/s, df = {fit['n_points']-2}, t_crit = {fit.get('t_crit', 0):.4f}")
                print(f"  95% CL = ± ({fit.get('t_crit', 0):.4f} * {fit.get('se_slope', 0):.4f}) = ± {fit['cl_95_cms']:.4f} cm/s\n")

    # Save and display summary table
    res_df = pd.DataFrame(results)
    res_df.to_csv('Exp2_Force_Velocity_Results.csv', index=False)
    print(res_df[['Trial_ID', 'Load_g', 'Force_dynes', 'Formatted_Velocity', 'R_squared', 'Notes']].to_string(index=False))

    # Print recommended 8 evenly-spaced points for Question 2 & MATLAB cftool (Question 3)
    recommended_ids = ['182007', '182608', '182758', '183134', '183741', '184023', '184242', '184557']
    sub_df = res_df[res_df['Trial_ID'].isin(recommended_ids)].copy()
    if not sub_df.empty:
        print("\n=== COPY-PASTE INTO MATLAB FOR QUESTION 3 (cftool) ===")
        print(f"Force = [{', '.join(f'{x:.0f}' for x in sub_df['Force_dynes'])}];")
        print(f"Velocity = [{', '.join(f'{x:.3f}' for x in sub_df['Velocity_cms'])}];")
        print("cftool")

if __name__ == '__main__':
    main('.')