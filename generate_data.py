import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_ecommerce_data(n_customers=1000, filename='ecommerce.csv'):
    """
    Generate synthetic e-commerce data with RFM metrics
    
    RFM Analysis:
    - Recency: Days since last purchase
    - Frequency: Number of purchases
    - Monetary Value: Total amount spent
    - Avg Purchase: Average purchase value
    """
    
    # Generate customer segments (for realistic distribution)
    segments = np.random.choice(['VIP', 'Regular', 'New', 'Dormant'], n_customers, p=[0.1, 0.4, 0.3, 0.2])
    
    data = []
    
    for i, segment in enumerate(segments):
        if segment == 'VIP':
            # VIP customers: Recent, frequent, high value
            recency = np.random.randint(1, 30)
            frequency = np.random.randint(20, 100)
            monetary = np.random.uniform(5000, 20000)
            
        elif segment == 'Regular':
            # Regular customers: Moderate on all metrics
            recency = np.random.randint(10, 60)
            frequency = np.random.randint(5, 25)
            monetary = np.random.uniform(500, 5000)
            
        elif segment == 'New':
            # New customers: Recent but low frequency
            recency = np.random.randint(1, 20)
            frequency = np.random.randint(1, 5)
            monetary = np.random.uniform(50, 1000)
            
        else:  # Dormant
            # Dormant customers: Not recent, variable frequency
            recency = np.random.randint(60, 365)
            frequency = np.random.randint(1, 15)
            monetary = np.random.uniform(100, 2000)
        
        # Add some noise to make it more realistic
        recency = max(1, recency + np.random.randint(-5, 5))
        frequency = max(1, frequency + np.random.randint(-2, 2))
        monetary = max(10, monetary * np.random.uniform(0.8, 1.2))
        
        # Calculate average purchase
        avg_purchase = monetary / frequency if frequency > 0 else monetary
        
        data.append({
            'CUSTOMER_ID': f'CUST_{i+1:05d}',
            'RECENCY': recency,
            'FREQUENCY': frequency,
            'MONETARY_VALUE': round(monetary, 2),
            'AVG_PURCHASE': round(avg_purchase, 2),
            'SEGMENT': segment  # Optional: for validation, can be removed
        })
    
    df = pd.DataFrame(data)
    
    # Add some missing values randomly (2% of data) to test preprocessing
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    missing_cols = np.random.choice(['RECENCY', 'FREQUENCY', 'MONETARY_VALUE', 'AVG_PURCHASE'], 
                                   size=len(missing_indices))
    
    for idx, col in zip(missing_indices, missing_cols):
        df.loc[idx, col] = np.nan
    
    return df

def generate_test_data(n_samples=100, filename='test_ecommerce.csv'):
    """
    Generate test data with same structure but different distribution
    """
    # Test data should have preprocessed format (scaled features)
    # Generate raw data first then scale it
    
    test_data = []
    
    for i in range(n_samples):
        # Random values in reasonable ranges
        recency_scaled = np.random.uniform(-2, 2)
        frequency_scaled = np.random.uniform(-2, 2)
        monetary_scaled = np.random.uniform(-2, 2)
        avg_purchase_scaled = np.random.uniform(-2, 2)
        
        test_data.append({
            'RECENCY': recency_scaled,
            'FREQUENCY': frequency_scaled,
            'MONETARY_VALUE': monetary_scaled,
            'AVG_PURCHASE': avg_purchase_scaled
        })
    
    df_test = pd.DataFrame(test_data)
    return df_test

# Create the data directory structure
def setup_data_files(base_path='Airflow_labs/lab1/dags/data'):
    """
    Create data files in the correct directory structure
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    train_df = generate_ecommerce_data(n_customers=1000)
    train_path = os.path.join(base_path, 'ecommerce.csv')
    train_df.to_csv(train_path, index=False)
    print(f"Training data saved to: {train_path}")
    print(f"Training data shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    print("\nFirst few rows:")
    print(train_df.head())
    
    # Generate test data (preprocessed format)
    print("\n" + "="*50)
    print("Generating test data...")
    test_df = generate_test_data(n_samples=100)
    test_path = os.path.join(base_path, 'test_ecommerce.csv')
    test_df.to_csv(test_path, index=False)
    print(f"Test data saved to: {test_path}")
    print(f"Test data shape: {test_df.shape}")
    print("\nFirst few rows:")
    print(test_df.head())
    
    # Generate data statistics
    print("\n" + "="*50)
    print("Data Statistics (Training):")
    print(train_df[['RECENCY', 'FREQUENCY', 'MONETARY_VALUE', 'AVG_PURCHASE']].describe())
    
    if 'SEGMENT' in train_df.columns:
        print("\nCustomer Segment Distribution:")
        print(train_df['SEGMENT'].value_counts())
    
    return train_df, test_df

if __name__ == "__main__":
    # If you're in the Airflow_labs/lab1/dags directory, use:
    # setup_data_files('data')
    
    # If you're in a different directory, adjust the path:
    train_df, test_df = setup_data_files('data')
    
    print("\n" + "="*50)
    print("✅ Dataset generation complete!")
    print("\nFiles created:")
    print("  - data/ecommerce.csv (training data)")
    print("  - data/test_ecommerce.csv (test data)")
    print("\nYour pipeline is now ready to run!")
    
    # Optional: Save without segment column for production
    # train_df_no_segment = train_df.drop('SEGMENT', axis=1)
    # train_df_no_segment.to_csv('data/ecommerce.csv', index=False)