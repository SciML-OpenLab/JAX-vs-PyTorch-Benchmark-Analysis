import json
import os
import matplotlib.pyplot as plt

def extract_stats(section_data):
    frameworks = []
    means = []
    stds = []

    for framework, stats in section_data.items():
        frameworks.append(framework)
        means.append(stats.get("mean_time", 0))
        stds.append(stats.get("std_time", 0))

    return frameworks, means, stds

def plot_section(name, section_data, save_dir="results"):
    frameworks, means, stds = extract_stats(section_data)

    plt.figure()
    plt.bar(frameworks, means, yerr=stds, capsize=5)
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"{name.upper()} Benchmark")

    filename = os.path.join(save_dir, f"{name}_plot.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

def plot_results(path="results/final_results.json"):
    with open(path) as f:
        data = json.load(f)

    # Handle logger structure
    if "results" in data:
        data = data["results"]

    os.makedirs("results", exist_ok=True)

    # If nested (multiple experiments)
    for section_name, section_data in data.items():
        if isinstance(section_data, dict):
            plot_section(section_name, section_data)

    print(" All plots generated!")
