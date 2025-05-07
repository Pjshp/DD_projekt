import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm
import math

# ---- Distance Functions ----
def geo_distance(coord1, coord2):
    """
    Calculates the geographical distance between two coordinates using the GEO distance formula.
    Used when EDGE_WEIGHT_TYPE is GEO.

    Args:
        coord1, coord2: Tuples representing (latitude, longitude)

    Returns:
        Distance in kilometers.
    """
    def to_radians(degrees):
        return degrees * math.pi / 180

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1, lon1 = to_radians(lat1), to_radians(lon1)
    lat2, lon2 = to_radians(lat2), to_radians(lon2)

    RRR = 6378.388  # Earth radius in km (used in TSPLIB conventions)
    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    return RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3))

# ---- TSP File Parser ----
def parse_tsp_file(filepath):
    """
    Parses a TSPLIB .tsp file and extracts node coordinates and the EDGE_WEIGHT_TYPE.

    Returns:
        coords: NumPy array of (x, y) or (latitude, longitude)
        edge_weight_type: e.g., "EUC_2D" or "GEO"
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    coords_section = False
    edge_weight_type = None
    coords = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "EDGE_WEIGHT_TYPE" in line:
            edge_weight_type = line.split(":")[1].strip()
        if "NODE_COORD_SECTION" in line:
            coords_section = True
            continue
        if "EOF" in line:
            break
        if coords_section:
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))

    return np.array(coords), edge_weight_type

# ---- Distance Selector ----

def euclidean(a, b):
    return np.linalg.norm(a - b)

def calculate_distance(coord1, coord2, edge_weight_type):
    """
    Chooses the correct distance metric based on edge_weight_type.
    """
    if edge_weight_type == "GEO":
        return geo_distance(coord1, coord2)
    else:
        return euclidean(coord1, coord2)

# ---- Training Data Generation ----

def generate_training_data(coords, edge_weight_type, n_samples=1000, limit_candidates_per_step=10):
    """
    Generates training samples for a classifier to learn greedy decisions in TSP.

    Each sample is a decision point where the model must choose the nearest city.
    """
    X, y = [], []
    n = len(coords)
    for _ in tqdm(range(n_samples), desc="Generating training data"):
        current = np.random.randint(n)
        visited = set([current])
        current_coords = coords[current]
        for _ in range(min(n - 1, 30)):  # limit steps to avoid long paths
            candidates = [i for i in range(n) if i not in visited]
            if not candidates:
                break
            np.random.shuffle(candidates)
            candidates = candidates[:min(limit_candidates_per_step, len(candidates))]

            dists = [calculate_distance(current_coords, coords[i], edge_weight_type) for i in candidates]
            min_idx = np.argmin(dists)
            target = candidates[min_idx]
            for i, dist in zip(candidates, dists):
                features = [current_coords[0], current_coords[1], coords[i][0], coords[i][1], dist]
                X.append(features)
                y.append(1 if i == target else 0)
            visited.add(target)
            current = target
            current_coords = coords[current]
    return np.array(X), np.array(y)

# ---- Model Training ----

def build_cart_model(X, y):
    """
    Trains a CART (decision tree) classifier to learn TSP decisions.
    """
    clf = DecisionTreeClassifier(max_depth=8)
    clf.fit(X, y)
    return clf

# ---- TSP Solver Using Model ----

def solve_tsp_with_cart(coords, clf, edge_weight_type):
    """
    Solves the TSP using greedy steps guided by a trained decision tree model.
    """
    n = len(coords)
    visited = set()
    path = []
    current = 0
    visited.add(current)
    path.append(current)
    pbar = tqdm(total=n - 1, desc="Solving TSP")
    while len(visited) < n:
        current_coords = coords[current]
        candidates = [i for i in range(n) if i not in visited]
        if not candidates:
            break
        features = []
        for i in candidates:
            dist = calculate_distance(current_coords, coords[i], edge_weight_type)
            features.append([current_coords[0], current_coords[1], coords[i][0], coords[i][1], dist])
        probs = clf.predict_proba(features)
        best_idx = np.argmax([p[1] for p in probs])
        next_city = candidates[best_idx]
        visited.add(next_city)
        path.append(next_city)
        current = next_city
        pbar.update(1)
    pbar.close()
    return path

# ---- Utility Function ----

def calculate_total_distance(coords, path, edge_weight_type):
    """
    Calculates the total distance of a TSP tour, including the return to the starting city.
    """
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += calculate_distance(coords[path[i]], coords[path[i+1]], edge_weight_type)
    # Add distance from last city to the first one to complete the cycle
    total_distance += calculate_distance(coords[path[-1]], coords[path[0]], edge_weight_type)
    return total_distance

# ---- Plotting ----

def plot_tsp_path(coords, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ordered_coords = coords[path + [path[0]]]
    ax.plot(ordered_coords[:, 0], ordered_coords[:, 1], marker='o')
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=6)
    ax.set_title("TSP Path (CART-guided)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    return fig

# ---- GUI App ----
class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver with CART")
        self.root.geometry("800x600")

        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=20)

        self.load_button = tk.Button(self.frame, text="Load TSP File", command=self.load_file)
        self.load_button.pack()

        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TSP Files", "*.tsp")])
        if file_path:
            try:
                coords, edge_weight_type = parse_tsp_file(file_path)
                print(f"[*] Loaded {len(coords)} cities.")
                
                # Generate training data
                X, y = generate_training_data(coords, edge_weight_type)

                # Train the model
                clf = build_cart_model(X, y)

                # Solve the TSP
                path = solve_tsp_with_cart(coords, clf, edge_weight_type)

                # Calculate the total distance of the path
                total_distance = calculate_total_distance(coords, path, edge_weight_type)

                # Show the total distance in the result label
                self.result_label.config(text=f"Total Distance: {total_distance:.2f}")

                # Plot the result
                fig = plot_tsp_path(coords, path)
                self.display_plot(fig)
            except Exception as e:
                messagebox.showerror("Error", f"Error while processing the file: {e}")

    def display_plot(self, fig):
        # Clear previous plot (if any)
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Display the new plot
        canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ---- Run the App ----
if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()
