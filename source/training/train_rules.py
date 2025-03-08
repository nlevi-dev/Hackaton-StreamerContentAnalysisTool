import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import plot_tree, export_text
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os

# Define save path
save_path = "./frontend/"
os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

# Load Data
df = pd.read_csv("/mnt-persist/data/merged_video_labels.csv")
df.dropna(how='any', inplace=True)

# Convert engagement score into binary categories: Low (0) and High (1)
df['engagement_category'] = pd.cut(df['score'], bins=[-1, 0.5, 1], labels=['low', 'high'])

# Drop original engagement score and unrelated columns
df = df.drop(columns=['score', 'video_number', 'timestamp'])

# Define features and target
X = df.drop(columns=['engagement_category'])
y = df['engagement_category']

# Handle class imbalance (Oversample High Engagement)
df_high = df[df['engagement_category'] == 'high']
df_low = df[df['engagement_category'] == 'low']
df_high_upsampled = resample(df_high, replace=True, n_samples=len(df_low), random_state=42)
df_balanced = pd.concat([df_low, df_high_upsampled]).sample(frac=1, random_state=42)

# Features & Target after balancing
X = df_balanced.drop(columns=['engagement_category'])
y = df_balanced['engagement_category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=2, min_samples_split=5, random_state=42)
tree.fit(X_train, y_train)

# Print tree rules
tree_rules = export_text(tree, feature_names=list(X.columns))
print(tree_rules)

# Feature Importance Analysis
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': tree.feature_importances_})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
print("Top 3 features driving high engagement:")
print(feature_importance.head(3))

# Save top 3 most important features to a file
top_features_path = os.path.join(save_path, "top_features.csv")
feature_importance.head(3).to_csv(top_features_path, index=False)