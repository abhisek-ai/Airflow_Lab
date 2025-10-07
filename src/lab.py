import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import pickle
import os
import numpy as np

def load_data():
    """
    Loads e-commerce data from CSV file and returns serialized data
    """
    # Using e-commerce dataset instead of credit card data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/ecommerce.csv"))
    serialized_data = pickle.dumps(df)
    return serialized_data
    
def data_preprocessing(data):
    """
    Performs preprocessing with RobustScaler and feature engineering
    """
    df = pickle.loads(data)
    df = df.dropna()
    
    # Different features: RFM (Recency, Frequency, Monetary) analysis
    clustering_data = df[["RECENCY", "FREQUENCY", "MONETARY_VALUE", "AVG_PURCHASE"]]
    
    # Using RobustScaler instead of MinMaxScaler (better for outliers)
    robust_scaler = RobustScaler()
    clustering_data_scaled = robust_scaler.fit_transform(clustering_data)
    
    clustering_serialized_data = pickle.dumps(clustering_data_scaled)
    return clustering_serialized_data

def build_save_model(data, filename):
    """
    Builds KMeans model with silhouette analysis for optimization
    """
    df = pickle.loads(data)
    
    # Different parameters and range
    kmeans_kwargs = {
        "init": "k-means++",  # Changed from random
        "n_init": 15,         # Increased iterations
        "max_iter": 500,      # More iterations
        "random_state": 2024  # Different seed
    }
    
    sse = []
    silhouette_scores = []
    
    # Testing fewer clusters (2-20 instead of 1-50)
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        
        # Add silhouette score calculation
        score = silhouette_score(df, kmeans.labels_)
        silhouette_scores.append(score)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'wb') as f:
        pickle.dump(kmeans, f)
    
    # Return both SSE and silhouette scores
    return {"sse": sse, "silhouette": silhouette_scores}

def load_model_elbow(filename, metrics):
    """
    Loads model and uses both elbow and silhouette methods for evaluation
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, 'rb'))
    
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test_ecommerce.csv"))
    
    # Elbow method
    kl = KneeLocator(
        range(2, 21), metrics["sse"], curve="convex", direction="decreasing"
    )
    
    # Best silhouette score
    best_silhouette = np.argmax(metrics["silhouette"]) + 2
    
    print(f"Elbow method optimal clusters: {kl.elbow}")
    print(f"Silhouette method optimal clusters: {best_silhouette}")
    
    predictions = loaded_model.predict(df)
    
    return {
        "prediction": predictions[0],
        "elbow_clusters": kl.elbow,
        "silhouette_clusters": best_silhouette
    }