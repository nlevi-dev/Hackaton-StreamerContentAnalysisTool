
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

df = pd.read_csv("/mnt-persist/data/merged_video_labels.csv", )

df.dropna(how='any', inplace=True)

# Convert engagement score into categories (adjust bins as needed)
df['engagement_category'] = pd.qcut(df['score'], q=3, labels=['low', 'medium', 'high'])

# Drop original engagement score
df = df.drop(columns=['score', 'video_number', 'timestamp'])

# Define features and target
X = df.drop(columns=['engagement_category'])
y = df['engagement_category']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=2, random_state=42)  # Adjust max_depth for interpretability
tree.fit(X_train, y_train)



# Extract rules
tree_rules = export_text(tree, feature_names=list(X.columns))
print(tree_rules)


plt.figure(figsize=(12,6))
plot_tree(tree, feature_names=X.columns, class_names=['low', 'medium', 'high'], filled=True)
plt.savefig("TREE.jpg")
plt.show()
