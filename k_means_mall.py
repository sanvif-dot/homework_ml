import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('howmework_MSanviF/data/Mall_Customers.csv')

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(df[['Income', 'Spending']])

# Find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Plot elbow curve
import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Fit K-Means model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to data
df['Cluster'] = y_kmeans

# Analyze characteristics of each cluster
cluster_means = df.groupby('Cluster').mean()
print(cluster_means)
