import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv("FC-test_Unique_Seq.csv")

# Strip leading/trailing whitespace from 'Sequence' and 'Amino Acid' columns
df['Sequence'] = df['Sequence'].str.strip()
df['Amino Acid'] = df['Amino Acid'].str.strip()

# Define initial and end sequences
initial = 'GGAGGCTCTCGGGACGAC'
end = 'GTCGTCCCGCCTTTAGGATTTACAG'
min_len = len(initial) + len(end)

# Filter the DataFrame to sequences matching the criteria
filtered_df = df[df['Sequence'].apply(lambda x: x.startswith(initial) and x.endswith(end) and len(x) > min_len)]
filtered_df.loc[: , 'Miss sequence'] = filtered_df['Sequence'].apply(lambda x: x[len(initial):-len(end)])

# Create separate DataFrames for N35 and N36 sequences
n35_df = filtered_df[filtered_df['Sequence'].apply(lambda x: len(x) - len(initial) - len(end) == 35)]
n36_df = filtered_df[filtered_df['Sequence'].apply(lambda x: len(x) - len(initial) - len(end) == 36)]

# Calculate the total counts for N35 and N36
total35 = n35_df['Count'].sum()
total36 = n36_df['Count'].sum()

# Write N35 and N36 sequences to CSV files
n35_df.to_csv('N35_sequencing.csv', index=False)
n36_df.to_csv('N36_sequencing.csv', index=False)

# Get the data for similarity calculation
data = n35_df['Sequence'].tolist()


# Calculate the similarity scores using NumPy 
num_strings = len(data)


similarity_matrix = np.zeros((num_strings, num_strings))


#Calculate the similarity scores on single core
for i in range(num_strings):
    for j in range(i, num_strings):
        similarity_score = fuzz.ratio(data[i], data[j])
        similarity_matrix[i][j] = similarity_score
        

#Converting similarity score matrix to a DataFrame
similarity_matrix = pd.DataFrame(similarity_matrix.T)

# Create a 2D contour plot using Seaborn with the mask for the upper triangle
# Plot similarity matrix WITHOUT diagonal data
#------------------------------------------------------------------------------
mask_ut=(np.triu(np.ones(similarity_matrix.shape)).astype(bool))
sns.heatmap(similarity_matrix, mask = mask_ut, cmap='flare', xticklabels='auto', yticklabels='auto', cbar=True, robust=True)

#Plot boxplot
#------------------------------------------------------------------------------

# Plot similarity matrix WITH diagonal data
#------------------------------------------------------------------------------
#df_lt = similarity_matrix.where(np.tril(np.ones(similarity_matrix.shape)).astype(bool))
#sns.heatmap(df_lt, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar=True)
#------------------------------------------------------------------------------

plt.title('Similarity scores of N-35 sequences')
plt.xlabel('Index number of sequence in Excell file') 
plt.ylabel('Index number of sequence in Excell file')
plt.show()
plt.savefig('N35_Similarity.png')

# Print summary information
print(f'Out of {len(data)} sequences, {len(n36_df)} have match sequence')
print('N35 =', total35)
print('N36 =', total36)


###############################################################################
#         Calculate the similarity scores in parallel                         #
#  For using this approach replace the following section with lines 41 to 44  #
###############################################################################
'''
# Define a function to calculate similarity between two strings

def calculate_similarity(i, j, data):
    similarity_score = fuzz.ratio(data[i], data[j])
    return i, j, similarity_score

# Calculate the similarity scores in parallel
results = Parallel(n_jobs=2)(
    delayed(calculate_similarity)(i, j) for i in range(num_strings) for j in range(i, num_strings)
)
'''
###############################################################################