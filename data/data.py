import pandas as pd

# Load Excel file without using the first row as header
df = pd.read_excel('Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx', header=None)

# Set custom header names
df.columns = ['Gardiner Code', 'Hieroglyph', 'Description', 'Details'] + [f'Extra_{i}' for i in range(len(df.columns) - 4)]

# Drop the first row (original header or unwanted data)
df = df.drop(0).reset_index(drop=True)

# Keep only the first four columns
df = df[['Gardiner Code', 'Hieroglyph', 'Description', 'Details']]

# Save the cleaned DataFrame to the same directory with the same name
df.to_excel('Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx', index=False)

print(df.head())