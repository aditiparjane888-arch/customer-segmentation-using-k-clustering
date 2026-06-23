#Customer Segmentation using K-Means Clustering customer-segmentation-using-k-clustering import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Sample Customer Data
data = {
    'Age': [25, 30, 22, 35, 40, 28, 50, 48, 33, 27],
    'Annual_Income': [30000, 40000, 25000, 60000, 80000,
                      35000, 90000, 85000, 50000, 32000],
    'Spending_Score': [80, 60, 90, 40, 20, 75, 15, 10, 50, 85]
}
df = pd.DataFrame(data)
X = df[['Annual_Income', 'Spending_Score']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
print(df)
plt.figure(figsize=(8,6))
plt.scatter(df['Annual_Income'], df['Spending_Score'],
            c=df['Cluster'], cmap='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            color='red', marker='X', s=200,
            label='Centroids')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()