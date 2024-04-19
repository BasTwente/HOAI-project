import pandas as pd

# List for storing csvs
files = ["poses0.csv","poses1.csv","poses2.csv", "poses3.csv","poses4.csv","poses5.csv","poses6.csv","poses7.csv", "posest.csv"]

# List for storing dataframes
dfs = []

# Iterate through each csv file
for filename in files:
    # Read csv file
    df = pd.read_csv(filename)
    # Append dataframe to the list
    dfs.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv("combined_data.csv", index=False)

print("CSV file was saved")