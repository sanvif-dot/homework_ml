# homework_ml

In this homework the bacground story is

"A mall wants to do customer analysis by categorizing customers based on the Income and Spending they spend while shopping. But the Mall doesn't know how many categories to use, can you help the Mall to do this analysis using machine learning."

This is the sample data
![image](https://user-images.githubusercontent.com/122470555/229262542-d9e0bba6-6d1d-4359-8e10-a532a85319fa.png)

One of the most popular machine learning algorithms for this type of analysis is clustering, which can group similar customers based on their characteristics.

To perform clustering, we will use the K-Means algorithm, which is a popular clustering algorithm. K-Means is an unsupervised learning algorithm that groups similar data points together. It works by assigning each data point to the nearest centroid (cluster center), and then moving the centroid to the mean of all the data points assigned to it. This process is repeated until the centroids no longer move, or a maximum number of iterations is reached.

To use K-Means for this problem, we first need to preprocess the data by normalizing the Income and Spending values. We can do this using the StandardScaler class from the scikit-learn library.

Next, we will select the number of clusters (categories) we want to use. We can do this by trying different numbers of clusters and evaluating the results using a metric like the within-cluster sum of squares (WCSS) or the silhouette score. WCSS measures the sum of squared distances between each point and its assigned centroid, while the silhouette score measures how well-separated the clusters are.

Once we have determined the optimal number of clusters, we can fit the K-Means model to the data and predict the cluster assignments for each customer. We can then analyze the characteristics of each cluster to gain insights into the different types of customers shopping at the mall.

![image](https://user-images.githubusercontent.com/122470555/229262811-8c8c5c73-08a8-462c-9b23-34b8cdb9b232.png)

In this code, we load the customer data into a Pandas DataFrame and preprocess the Income and Spending values using the StandardScaler class. We then iterate through different numbers of clusters and use the K-Means algorithm to fit a model for each number of clusters, recording the within-cluster sum of squares for each model. We plot these values to visualize the elbow curve and determine the optimal number of clusters (in this case, 5).

We then fit the K-Means model to the data with 5 clusters and add the cluster labels to the original DataFrame. Finally, we group the data by cluster and compute the mean values for each variable, which gives us insights into the characteristics of each cluster.

The output of the above code will show the mean values for each variable (Age, Income, and Spending) for each of the 5 clusters.
