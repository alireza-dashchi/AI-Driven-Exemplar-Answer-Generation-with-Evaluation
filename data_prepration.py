import json
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path, train_output='train_data.json', test_output='test_data.json', holdout_output='holdout_data.json', test_size=0.3, holdout_size=0.3):
    
    # Load the original data
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        # Save empty DataFrames if no data
        pd.DataFrame().to_json(train_output, orient='records', lines=True)
        pd.DataFrame().to_json(test_output, orient='records', lines=True)
        pd.DataFrame().to_json(holdout_output, orient='records', lines=True)
        print("Input data is empty. Created empty train, test, and holdout files.")
        return

    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=100)
    test_df, holdout_df = train_test_split(test_df, test_size=holdout_size, random_state=100)

    # Save them
    train_df.to_json(train_output, orient='records', lines=True)
    test_df.to_json(test_output, orient='records', lines=True)
    holdout_df.to_json(holdout_output, orient='records', lines=True)
    print(f"Data split completed. Training data: {train_output}, testing data: {test_output}, holdout data: {holdout_output}.")
