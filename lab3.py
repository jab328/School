"""Analyze the three Lab 3 tensile tests using the documented window rule.

Run: python lab3.py "junior's data.csv"
"""

import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_from_file(filename, sample_length, sample_diameter):
    """Return strain and stress (Pa) from five-column data; sizes in mm."""
    table = pd.read_csv(filename, encoding="utf-8-sig")
    if table.shape[1] != 5:
        raise ValueError("Expected one five-column material dataset.")
    values = table.iloc[:, [2, 3]].apply(
        pd.to_numeric, errors="coerce").dropna().to_numpy()
    if sample_length <= 0 or sample_diameter <= 0 or len(values) < 2:
        raise ValueError("Positive dimensions and valid samples required.")
    area = np.pi * (sample_diameter * 1e-3)**2 / 4
    strain = values[:, 0] / (sample_length * 1e-3)
    stress = values[:, 1] / area
    return strain, stress

def find_yield_index(strain, stress):
    """Return the index where yield is detected using the Lab 3 protocol window rule."""
    strain, stress = np.asarray(strain), np.asarray(stress)
    
    window_size = 50
    window_shift = 10
    previous_slope = 0.0
    
    num_windows = (len(strain) - window_size) // window_shift + 1
    for window_index in range(num_windows):
        i = window_index * window_shift
        current_slope = float(
            np.polyfit(strain[i : i + window_size], stress[i : i + window_size], 1)[0]
        )
        if window_index > 0 and current_slope < 0.95 * previous_slope:
            return i
        previous_slope = current_slope
        
    return len(strain) - 1
    

def find_elastic_modulus(strain, stress, yield_index):
    """Return the pre-yield least-squares slope in Pa, fitting an intercept."""
    if not 2 <= yield_index < len(strain):
        raise ValueError("Yield index must leave at least two fit samples.")
    return float(np.polyfit(strain[:yield_index],
                            stress[:yield_index], 1)[0])


def analyze_combined(filename):
    """Analyze the 15-column export as three five-column material blocks."""
    table = pd.read_csv(filename, encoding="utf-8-sig")
    if table.shape[1] != 15:
        raise ValueError("Expected a combined 15-column export.")
    # Name, block start, length (mm), diameter (mm), Eref (GPa), UTSref (MPa).
    specs = [("Brass 360", 0, 37.69, 3.38, 80., 500.),
             ("Annealed steel", 5, 37.63, 3.27, 200., 400.),
             ("Polyethylene", 10, 37.00, 3.34, 1., 30.)]
    results = {}
    for name, start, length, diameter, ref_e, ref_s in specs:
        block = io.StringIO(table.iloc[:, start:start + 5].to_csv(index=False))
        strain, stress = data_from_file(block, length, diameter)
        iy = find_yield_index(strain, stress)
        modulus = find_elastic_modulus(strain, stress, iy)
        iu = int(np.argmax(stress))
        results[name] = dict(strain=strain, stress=stress, yield_index=iy,
            modulus=modulus, peak_index=iu, max_strain=float(np.max(strain)),
            e_deviation=100 * (modulus / (ref_e * 1e9) - 1),
            s_deviation=100 * (stress[iu] / (ref_s * 1e6) - 1))
    return results


if __name__ == "__main__":
    results = analyze_combined(sys.argv[1])
    
    # 1. Print terminal metrics
    for name, r in results.items():
        iy, iu = r["yield_index"], r["peak_index"]
        print(name, "yield index:", iy,
              "E (GPa):", round(r["modulus"] / 1e9, 2),
              "yield (MPa):", round(r["stress"][iy] / 1e6, 2),
              "max stress (MPa):", round(r["stress"][iu] / 1e6, 2),
              "E deviation (%):", round(r["e_deviation"], 2))
              
    # 2. Generate and save the stress-strain plot
    plt.figure(figsize=(10, 6))
    for name, r in results.items():
        strain = r["strain"]
        stress_mpa = r["stress"] / 1e6  # Convert Pa to MPa for plotting
        iy, iu = r["yield_index"], r["peak_index"]
        
        # Plot the main curve
        line, = plt.plot(strain, stress_mpa, label=name)
        color = line.get_color()
        
        # Mark Yield point
        plt.plot(strain[iy], stress_mpa[iy], 'o', color=color)
        # Mark Ultimate Tensile Strength
        plt.plot(strain[iu], stress_mpa[iu], 'x', color=color)
        
    plt.title("Engineering Stress-Strain Curves")
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plot_filename = "stress_strain_curves.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot saved successfully to '{plot_filename}'")