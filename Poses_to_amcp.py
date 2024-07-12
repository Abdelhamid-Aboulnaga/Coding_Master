import pandas as pd
import sys 

"""To run this code use: $ python filename.py PATH_TO_top_poses.scores PATH_TO_smiles_repository.smi New_output.txt sample_of_2millions.txt
"""
"""This is a file that takes the output scores of docking and match it to their corresponding SMILES in a different file by matching the molecular ID. If the molecular ID doesn't have a docked pope, then we give it a very bad score
"""

def preprocess_file1(file1):
    # Read file1 into a pandas DataFrame
    df1 = pd.read_csv(file1, sep='\s+', skiprows=1, dtype='str', header=None)
    
    # Split column 0 by dot (.) and keep only the first part
    df1[0] = df1[0].str.split('.').str[0]
    df1_unique = df1.groupby([0]).first().reset_index()


    # Rename columns for clarity
    df1_unique.columns = ['col1_file1', 'col2_file1', 'col3_file1', 'col4_file1', 'col5_file1', 'col6_file1', 'col7_file1', 'col8_file1', 'col9_file1', 'col10_file1', 'col11_file1', 'col12_file1', 'col13_file1', 'col14_file1', 'col15_file', 'col16_file1', 'col17_file1']  # assuming 17 columns in file1                    
    
    return df1_unique

def merge_files(file1, file2, output_file, sample_2m):
    # Preprocess file1
    df1 = preprocess_file1(file1)
    # Read file2 into a pandas DataFrame, selecting only necessary columns
    df2 = pd.read_csv(file2, sep='\s+', dtype='str', header=None, usecols=[0, 1])  # read only needed columns from file2
    df2.columns = ['col1_file2', 'col2_file2']  # assuming 2 columns in file2
    # Perform inner join based on col1_file1 and col2_file2
    merged_df = pd.merge(df1, df2, left_on='col1_file1', right_on='col2_file2', how='inner')

    # Select columns for output
    result_df = merged_df[['col1_file2', 'col1_file1', 'col17_file1']]
    
    # Identify non-matching values from col2_file2 to col1_file1
    non_matching_df = df2[~df2['col2_file2'].isin(df1['col1_file1'])]

    # Create new rows for non-matching values
    non_matching_df['col3'] = '100'
    non_matching_df = non_matching_df[['col1_file2', 'col2_file2', 'col3']]
    non_matching_df.columns = ['col1_file2', 'col1_file1', 'col17_file1']

    # Append non-matching rows to the result DataFrame
    result_df = pd.concat([result_df, non_matching_df], ignore_index=True)
    result_df['col17_file1'] = result_df['col17_file1'].astype(float)
    
    # Write result to output file
    result_df.to_csv(output_file, sep=' ', header=False, index=False)
    
    # Sort the dataframe (assuming higher values are "better")
    result_df = result_df.sort_values(by='col17_file1', ascending=True).reset_index(drop=True)

    # Calculate the number of rows
    total_rows = len(result_df)
    top_1_percent_size = int(0.01 * total_rows)
    bottom_99_percent_size = total_rows - top_1_percent_size

    # Split into top 1% and bottom 99%
    top_1_percent_df = result_df.iloc[:top_1_percent_size]
    bottom_99_percent_df = result_df.iloc[top_1_percent_size:]

    # Calculate the sample sizes
    top_sample_size = int(0.01 * 2000000)
    bottom_sample_size = 2000000 - top_sample_size

    # Sample 1% of top 1% and 99% of bottom 99%
    top_sample = top_1_percent_df.sample(n=top_sample_size, random_state=1)
    bottom_sample = bottom_99_percent_df.sample(n=bottom_sample_size, random_state=1)

    # Combine the samples
    df_dock_subset = pd.concat([top_sample, bottom_sample]).reset_index(drop=True)

    df_dock_subset.to_csv(sample_2m, sep= ' ', header=False, index=False)    

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <file1> <file2> <output_file> <sample_2m>")
    else:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        output_file = sys.argv[3]
        sample_2m = sys.argv[4]
        merge_files(file1, file2, output_file, sample_2m)
