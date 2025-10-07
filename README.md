# Airflow Lab 1: E-commerce Customer Segmentation Pipeline

## 📋 Overview
Modified Apache Airflow pipeline that performs **customer segmentation using RFM analysis** instead of credit card fraud detection.

## 🔄 Key Changes
- **Dataset**: E-commerce RFM metrics (Recency, Frequency, Monetary)
- **Algorithm**: KMeans clustering with elbow & silhouette optimization  
- **Preprocessing**: RobustScaler instead of MinMaxScaler
- **Schedule**: Weekly instead of daily
- **Evaluation**: Dual optimization (elbow method + silhouette score)


## 🚀 Pipeline Tasks
1. `load_ecommerce_data` - Load customer transaction data
2. `rfm_preprocessing` - Apply RobustScaler and feature engineering  
3. `train_clustering_model` - Train KMeans (2-20 clusters)
4. `evaluate_model` - Find optimal clusters using elbow & silhouette

## 📊 Results
- **Optimal Clusters**: 4-6 (Elbow), 4 (Silhouette)
- **Segments Identified**: VIP, Regular, New, Dormant customers

## 🛠️ Tech Stack
- Apache Airflow 3.x
- Python 3.13
- scikit-learn, pandas, kneed

## 📸 Screenshots
![Airflow DAG Graph](screenshots/dag_graph.png)
![Pipeline Execution](screenshots/execution.png)

## 👤 Author
Abhisek Mallick
