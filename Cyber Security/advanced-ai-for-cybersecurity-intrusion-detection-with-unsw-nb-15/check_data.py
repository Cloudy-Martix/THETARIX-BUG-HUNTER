import pandas as pd

# Load sample to see structure
train_df = pd.read_parquet('UNSW_NB15_training-set.parquet')
test_df = pd.read_parquet('UNSW_NB15_testing-set.parquet')

print("Train Columns:", train_df.columns.tolist())
print("\nTrain Sample:\n", train_df.head())
print("\nAttack Categories:", train_df['attack_cat'].unique())
print("\nLabel Distribution:\n", train_df['label'].value_counts())
