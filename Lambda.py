import json
import boto3
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, VARCHAR, TIME
from io import StringIO
import re
import os
import logging

# Initialize S3 client and setup logging
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define SNS topic ARN
sns_topic_arn = os.environ['sns_topic_arn']


def lambda_handler(event, context):
    try:
        # Retrieve bucket and file name from the event
        s3_bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
        s3_file_name = event["Records"][0]["s3"]["object"]["key"]
        
        # Get the file object from S3
        s3_object = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_file_name)
        body = s3_object['Body']
        csv_string = body.read().decode('utf-8')
        
        # Load CSV data into a DataFrame
        df = pd.read_csv(StringIO(csv_string))
        
        # Rename columns if necessary
        df.rename(columns={
            'state': 'state_code',
            'location': 'location_coord',
            'customer_ID': 'customer_id',
            'C_previous': 'c_previous'
        }, inplace=True)
        
        # Drop rows with null 'cost'
        df = df.dropna(subset=['cost'])
        
        # Function to drop rows with high null values
        def drop_rows_with_high_nulls(df, threshold_fraction=0.5):
            threshold = int(threshold_fraction * df.shape[1])
            return df.dropna(thresh=threshold)
        
        df = drop_rows_with_high_nulls(df, 0.5)
        
        # Function to drop invalid rows based on defined criteria
        def drop_invalid_rows(df):
            valid_record_type = [0, 1]
            valid_days = list(range(7))
            valid_homeowner = [0, 1]
            valid_car_value = list('abcdefghi')
            valid_risk_factor = [1, 2, 3, 4]
            valid_married_couple = [0, 1]
            valid_range_0_4 = [0, 1, 2, 3, 4]
            
            df = df[
                ((df['record_type'].isin(valid_record_type)) | df['record_type'].isnull()) &
                ((df['day'].isin(valid_days)) | df['day'].isnull()) &
                ((df['homeowner'].isin(valid_homeowner)) | df['homeowner'].isnull()) &
                ((df['car_value'].isin(valid_car_value)) | df['car_value'].isnull()) &
                ((df['risk_factor'].isin(valid_risk_factor)) | df['risk_factor'].isnull()) &
                ((df['married_couple'].isin(valid_married_couple)) | df['married_couple'].isnull()) &
                ((df['c_previous'].isin(valid_range_0_4)) | df['c_previous'].isnull()) &
                ((df['A'].isin(valid_range_0_4)) | df['A'].isnull()) &
                ((df['B'].isin(valid_range_0_4)) | df['B'].isnull()) &
                ((df['C'].isin(valid_range_0_4)) | df['C'].isnull()) &
                ((df['D'].isin(valid_range_0_4)) | df['D'].isnull()) &
                ((df['E'].isin(valid_range_0_4)) | df['E'].isnull()) &
                ((df['F'].isin(valid_range_0_4)) | df['F'].isnull()) &
                ((df['G'].isin(valid_range_0_4)) | df['G'].isnull()) &
                ((df['age_youngest'] <= df['age_oldest']) | df['age_youngest'].isnull() | df['age_oldest'].isnull()) &
                ((df['state_code'].str.len() == 2) | df['state_code'].isnull())
            ]
            
            return df
        
        df = drop_invalid_rows(df)
        
        # Validate and clean the 'time' column
        def validate_time_format(time_str):
            if isinstance(time_str, str) and re.match(r'^(?:[01]\d|2[0-3]):[0-5]\d$', time_str):
                return time_str
            return None
        
        df['time'] = df['time'].apply(validate_time_format)
        
        # Database connection parameters
        db_host = os.environ['db_host']  
        db_port = os.environ['db_port']  
        db_name = os.environ['db_name']  
        db_user = os.environ['db_user']  
        db_password = os.environ['db_password']  
        
        # Create the database connection string
        db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        
        # Create engine and connect
        engine = create_engine(db_url)
        
        # Define table metadata
        metadata = MetaData()
        table_name = 'insurance_policy_data'
        
        columns = [
            Column('customer_id', Integer),
            Column('shopping_pt', Integer),
            Column('record_type', Integer),
            Column('day', Integer),
            Column('time', TIME),
            Column('state_code', VARCHAR(2)),
            Column('location_coord', Integer),
            Column('group_size', Integer),
            Column('homeowner', Integer),
            Column('car_age', Integer),
            Column('car_value', VARCHAR(2)),
            Column('risk_factor', Integer),
            Column('age_oldest', Integer),
            Column('age_youngest', Integer),
            Column('married_couple', Integer),
            Column('c_previous', Integer),
            Column('duration_previous', Integer),
            Column('A', Integer),
            Column('B', Integer),
            Column('C', Integer),
            Column('D', Integer),
            Column('E', Integer),
            Column('F', Integer),
            Column('G', Integer),
            Column('cost', Integer)
        ]
        
        # Create table if not exists
        table = Table(table_name, metadata, *columns)
        metadata.create_all(engine, checkfirst=True)
        
        # Insert data into the table
        with engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists='append', index=False, chunksize=1000)
        
        logger.info(f"Data inserted into {table_name} successfully")
    
    except KeyError as e:
        error_message = f"KeyError: {e}"
        logger.error(error_message)
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Subject='Lambda Function Error Notification',
            Message=error_message
        )
        raise e
    
    except Exception as e:
        error_message = f"An error occurred: {e}"
        logger.error(error_message)
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Subject='Lambda Function Error Notification',
            Message=error_message
        )
        raise e
    
    return {
        'statusCode': 200,
        'body': json.dumps('Data processed and inserted successfully.')
    }
    
# import json
# import boto3
# import pandas as pd
# from sqlalchemy import create_engine, MetaData, Table, Boolean
# from sqlalchemy import Column, Integer, VARCHAR, TIME
# from io import StringIO
# import re

# s3_client = boto3.client('s3')

# def lambda_handler(event, context):
#     try:
#         # Retrieve bucket and file name from the event
#         s3_bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
#         s3_file_name = event["Records"][0]["s3"]["object"]["key"]
        
#         # print('s3 file name:', s3_file_name)
        
#         # Get the file object from S3
#         s3_object = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_file_name)
#         # print('s3 object:', s3_object)
        
#         body = s3_object['Body']
#         csv_string = body.read().decode('utf-8')
        
#         # Load CSV data into a DataFrame
#         df = pd.read_csv(StringIO(csv_string))
        
#         # Rename columns if necessary
#         df.rename(columns={
#             'state': 'state_code',
#             'location': 'location_coord',
#             'customer_ID': 'customer_id',
#             'C_previous': 'c_previous'
#         }, inplace=True)
#         # print('columns renamed')
        
#         # Drop rows with null 'cost'
#         df = df.dropna(subset=['cost'])
#         # print('null cost dropped')
        
#         # Calculate the threshold for null values 
#         def drop_rows_with_high_nulls(df, threshold_fraction=0.5):
#             threshold = int(threshold_fraction * df.shape[1])
#             # Drop rows where the number of null values exceeds the threshold
#             df = df.dropna(thresh=threshold)
#             return df
            
#         df = drop_rows_with_high_nulls(df, 0.5)
#         # print('50% checked')
        
#         def drop_invalid_rows(df):
#             # Define valid ranges/values for columns
#             valid_record_type = [0, 1]
#             valid_days = list(range(7))
#             valid_homeowner = [0, 1]
#             valid_car_value = list('abcdefghi')
#             valid_risk_factor = [1, 2, 3, 4]
#             valid_married_couple = [0, 1]
#             valid_range_0_4 = [0, 1, 2, 3, 4]
        
#             # Apply sanity checks
#             df = df[
#                 ((df['record_type'].isin(valid_record_type)) | (df['record_type'].isnull())) &
#                 ((df['day'].isin(valid_days)) | (df['day'].isnull())) &
#                 ((df['homeowner'].isin(valid_homeowner)) | (df['homeowner'].isnull())) &
#                 ((df['car_value'].isin(valid_car_value)) | (df['car_value'].isnull())) &
#                 ((df['risk_factor'].isin(valid_risk_factor)) | (df['risk_factor'].isnull())) &
#                 ((df['married_couple'].isin(valid_married_couple)) | (df['married_couple'].isnull())) &
#                 ((df['c_previous'].isin(valid_range_0_4)) | (df['c_previous'].isnull())) &
#                 ((df['A'].isin(valid_range_0_4)) | (df['A'].isnull())) &
#                 ((df['B'].isin(valid_range_0_4)) | (df['B'].isnull())) &
#                 ((df['C'].isin(valid_range_0_4)) | (df['C'].isnull())) &
#                 ((df['D'].isin(valid_range_0_4)) | (df['D'].isnull())) &
#                 ((df['E'].isin(valid_range_0_4)) | (df['E'].isnull())) &
#                 ((df['F'].isin(valid_range_0_4)) | (df['F'].isnull())) &
#                 ((df['G'].isin(valid_range_0_4)) | (df['G'].isnull())) &
#                 ((df['age_youngest'] <= df['age_oldest']) | (df['age_youngest'].isnull()) | (df['age_oldest'].isnull())) &
#                 ((df['state_code'].str.len() == 2) | (df['state_code'].isnull()))
#             ]
            
#             return df
        
#         # Filter invalid rows
#         df = drop_invalid_rows(df)
#         # print('sanity checked')
        
        
#         # Validate and clean the 'time' column
#         def validate_time_format(time_str):
#             if isinstance(time_str, str) and re.match(r'^(?:[01]\d|2[0-3]):[0-5]\d$', time_str):
#                 return time_str
#             return None
        
#         df['time'] = df['time'].apply(validate_time_format)
#         # print('time column validated and cleaned')
        
        
#         # Print the first 3 rows of the DataFrame for debugging
#         print("\n\n", df.head(5))
        
        
#         # Define valid columns and their types for the database table
#         metadata = MetaData()
#         table_name = 'insurance_policy_data'
        
#         columns = [
#             Column('customer_id', Integer),
#             Column('shopping_pt', Integer),
#             Column('record_type', Integer),
#             Column('day', Integer),
#             Column('time', TIME),
#             Column('state_code', VARCHAR(2)),
#             Column('location_coord', Integer),
#             Column('group_size', Integer),
#             Column('homeowner', Integer),
#             Column('car_age', Integer),
#             Column('car_value', VARCHAR(2)),
#             Column('risk_factor', Integer),
#             Column('age_oldest', Integer),
#             Column('age_youngest', Integer),
#             Column('married_couple', Integer),
#             Column('c_previous', Integer),
#             Column('duration_previous', Integer),
#             Column('A', Integer),
#             Column('B', Integer),
#             Column('C', Integer),
#             Column('D', Integer),
#             Column('E', Integer),
#             Column('F', Integer),
#             Column('G', Integer),
#             Column('cost', Integer)
#         ]
        
#         # Database connection parameters
#         db_host = 'car-policy-db-instance.cz0a2quw2mzc.ap-south-1.rds.amazonaws.com'  # Change this to your RDS endpoint
#         db_port = '5432'                                                       # Default port for PostgreSQL
#         db_name = 'car_insurance_policy_db'                                    # Change this to your database name
#         db_user = 'postgres'                                                   # Change this to your database username
#         db_password = 'Gr0w3**Pa$$w0rd'                                        # Change this to your database password
        
#         # Create the database connection string
#         db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
#         # print('db url:', db_url)
        
#         # Create engine and connect
#         engine = create_engine(db_url)
#         # print('engine created')
        
#         # Create table if not exists
#         table = Table(table_name, metadata, *columns)
#         # print('table created')
        
#         metadata.create_all(engine, checkfirst=True)
#         # print('metadata created')
        
#         # Insert data into the table
#         try:
#             with engine.connect() as conn:
#                 df.to_sql(table_name, conn, if_exists='append', index=False, chunksize=1000)
#                 print(f"Data inserted into {table_name} successfully")
#         except Exception as e:
#             print(f"An error occurred: {e}")

    
    
#     except Exception as e:
#         print(f"Error: {e}")
    
    
#     # Return a response
#     return {
#         'statusCode': 200,
#         'body': json.dumps('Hello from Lambda!')
#     }
