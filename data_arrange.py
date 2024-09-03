import pandas as pd 
import numpy as np

df = pd.read_csv('final_filtered_merged_df_transposed.csv')
df_transposed = df.transpose()

df_transposed.columns = df_transposed.iloc[0]
df_transposed = df_transposed.drop(df_transposed.index[0]).reset_index(drop=True)
# List of columns to drop
columns_to_drop = [
    'PHC_MEM', 'PHC_EXF', 'PHC_LAN', 'PHC_VSP', 'update_stamp', 'PHC_AGE',
    'SUBJID', 'RID', 'PHASE', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'PHC_VISIT',
    'PHC_Sex', 'PHC_Education', 'PHC_Ethnicity', 'Phase', 'Visit', 'SubjectID'
]

# Drop the specified columns
df_transposed = df_transposed.drop(columns=columns_to_drop)

column_names = df_transposed.columns.tolist()
# Convert 'PHC_Race' to numeric, forcing errors to NaN
df_transposed['PHC_Race'] = pd.to_numeric(df_transposed['PHC_Race'], errors='coerce')
# Create final_df where 'PHC_Race' column contains only 5 or 3
final_df = df_transposed[df_transposed['PHC_Race'].isin([5, 3])]
# Convert all columns in final_df to numeric, forcing errors to NaN
final_df = final_df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values (if any) after conversion
final_df = final_df.dropna()
