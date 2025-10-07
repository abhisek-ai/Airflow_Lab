from airflow import DAG
from airflow.operators.python import PythonOperator  # FIXED: Updated import for Airflow 3.x
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# Note: In Airflow 3.x, XCom pickling is handled differently
# You may need to configure it in airflow.cfg

default_args = {
    'owner': 'abhisekai',  # Changed to your name
    'start_date': datetime(2025, 1, 20),
    'retries': 1,  # Added retry
    'retry_delay': timedelta(minutes=3),
    'email': ['abhisek.mallick08@gmail.com'],  # Your email
    'email_on_failure': True,
}

dag = DAG(
    'Ecommerce_Segmentation_Pipeline',  # Different DAG name
    default_args=default_args,
    description='E-commerce customer segmentation using RFM analysis',
    schedule_interval='@weekly',  # Different schedule
    catchup=False,
    tags=['ml', 'clustering', 'ecommerce', 'rfm'],  # Added tags
)

# Task definitions with improved documentation
load_data_task = PythonOperator(
    task_id='load_ecommerce_data',
    python_callable=load_data,
    dag=dag,
    doc_md="Load e-commerce customer transaction data"
)

data_preprocessing_task = PythonOperator(
    task_id='rfm_preprocessing',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
    doc_md="Preprocess data with RFM feature engineering"
)

build_save_model_task = PythonOperator(
    task_id='train_clustering_model',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "rfm_model.pkl"],  # Different filename
    provide_context=True,
    dag=dag,
    doc_md="Train KMeans with silhouette optimization"
)

load_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=load_model_elbow,
    op_args=["rfm_model.pkl", build_save_model_task.output],
    dag=dag,
    doc_md="Evaluate model using elbow and silhouette methods"
)

# Task dependencies
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

if __name__ == "__main__":
    dag.test()