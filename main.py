import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

FEATURES = ["longitude", "latitude"]
RANDOM_STATE = 42

# Load data
real = pd.read_csv("real.csv")
new_points = pd.read_csv("gold_prediction_new_points.csv")

X_train = real[FEATURES]
X_new = new_points[FEATURES]

print(f"Training samples: {len(real)}")
print(f"New points:       {len(new_points)}")

# ---------- Task 1: Regression ----------
regressor = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
regressor.fit(X_train, real["concentration_real"])
new_points["concentration_pred"] = regressor.predict(X_new)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(new_points["longitude"], new_points["latitude"],
                      c=new_points["concentration_pred"], cmap="viridis",
                      s=70, edgecolors="black", linewidths=0.5)
plt.colorbar(scatter, label="Predicted gold concentration")
plt.title("Random Forest Regression: Predicted Gold Concentration")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("regression_plot.png", dpi=150)
plt.close()

# ---------- Task 2: Classification ----------
# Low: c <= 1.52, Medium: 1.52 < c <= 2.02, High: c > 2.02
class_labels = ["Low", "Medium", "High"]
real["concentration_class"] = pd.cut(real["concentration_real"],
                                     bins=[float("-inf"), 1.52, 2.02, float("inf")],
                                     labels=class_labels)

print("\nActual class distribution (expected: Low 87, Medium 88, High 84):")
print(real["concentration_class"].value_counts().reindex(class_labels).to_string())

classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
classifier.fit(X_train, real["concentration_class"])
new_points["concentration_class_pred"] = classifier.predict(X_new)

class_colors = {"Low": "tab:blue", "Medium": "tab:orange", "High": "tab:red"}

plt.figure(figsize=(9, 6))
for label in class_labels:
    points = new_points[new_points["concentration_class_pred"] == label]
    plt.scatter(points["longitude"], points["latitude"], color=class_colors[label],
                label=label, s=70, edgecolors="black", linewidths=0.5)
plt.legend(title="Predicted class", loc="upper left", bbox_to_anchor=(1.02, 1))
plt.title("Random Forest Classification: Predicted Gold Concentration Class")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("classification_plot.png", dpi=150)
plt.close()

# Save results
new_points.to_csv("gold_predictions.csv", index=False)

print("\nRegressor parameters:", regressor.get_params())
print("\nPredicted class distribution:")
print(new_points["concentration_class_pred"].value_counts().reindex(class_labels).to_string())
print("\nPreview of predictions:")
print(new_points.head(10).to_string(index=False))
print("\nSaved: gold_predictions.csv, regression_plot.png, classification_plot.png")
