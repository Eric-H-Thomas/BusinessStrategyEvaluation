import math
import os
import pandas as pd
import numpy as np

def csv_to_dataframe(file_path, chunk_size=10000):
    """
    Reads a large CSV file in chunks and returns a concatenated DataFrame.

    Parameters:
    - file_path: str, path to the CSV file
    - chunk_size: int, number of rows per chunk (default is 10,000)

    Returns:
    - DataFrame: Pandas DataFrame containing the data
    """
    chunks = []
    try:
        # Read the file in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        # Concatenate all chunks into a single DataFrame
        data = pd.concat(chunks, axis=0)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    

def get_agent_statistics(df, firm_indices):
    # For each simulation-firm combination, get the ending capital value
    ending_capital_values = []
    last_time_step = df['Step'].max()
    num_simulations = df['Sim'].max() + 1
    
    for firm_id in firm_indices:
        for i in range(num_simulations):
            filtered_df = df[(df['Sim'] == i) &
                         (df['Step'] == last_time_step) &
                         (df['Firm'] == firm_id) &
                         (df['Market'] == 0)]

            if not filtered_df.empty:
                capital_value = filtered_df['Capital'].values[0]
                ending_capital_values.append(capital_value)
            else:
                print("Error in get_average_ending_capital(): No matching record for given sim, step, firm, and market.")
                

    # Convert the list to a NumPy array
    arr = np.array(ending_capital_values)
    
    # Determine which values are not close to 0 (these represent simulation-firm combinations that did not end in bankruptcy)
    not_close_to_zero = ~np.isclose(arr, 0, atol=1e-06)
    
    # Calculate the percentage of numbers not close to 0
    percent_not_bankrupt = np.sum(not_close_to_zero) / len(arr) * 100
    
    # Calculate the average of numbers not close to 0
    if np.any(not_close_to_zero):  # Check if there are any values not close to 0
        average_ending_capital_when_not_bankrupt = np.mean(arr[not_close_to_zero])
    else:
        average_ending_capital_when_not_bankrupt = 0.0
    
    return percent_not_bankrupt, average_ending_capital_when_not_bankrupt
    

def main():
    file_path_start = 'C:\\Users\\ARG\\Desktop\\BusinessStrategySimulator2Windows\\BusinessStrategySimulator2Windows\\BusinessSimulator2.0\\MasterOutputFiles\\MasterOutputMaxDemand'
    file_path_end = '.csv'
    file_paths = []
    for demand_intercept_max in range(6,31):
        file_paths.append(file_path_start + str(demand_intercept_max) + file_path_end)

    starting_capital_amount = 2000 # We have to supply this information since it isn't explicitly included in the output file
    firm_id_of_ai_agent_being_trained = [4]
    firm_ids_of_sophisticated_agents = [0,1]
    firm_ids_of_naive_agents = [2,3]

    for i in range(len(file_paths)):
        file_path = file_paths[i]
        dataframe = csv_to_dataframe(file_path)

        print("------------------------------------------------")
        print(f"Results for the data stored in {os.path.basename(file_path)}:\n")
        if dataframe is not None:
            # Analyze AI agent performance
            if firm_id_of_ai_agent_being_trained:
                percent_not_bankrupt, average_ending_capital_when_not_bankrupt = get_agent_statistics(dataframe, firm_id_of_ai_agent_being_trained)
                print(f"The AI agent avoided bankruptcy in {percent_not_bankrupt:.2f} percent of the simulations.")
                if not math.isclose(percent_not_bankrupt, 0.0, rel_tol=0.0, abs_tol=1e-06):
                    print(f"When it avoided bankruptcy, it finished with an average capital of {average_ending_capital_when_not_bankrupt:.2f}.")

            # Analyze sophisticated agent performance
            if firm_id_of_ai_agent_being_trained:
                percent_not_bankrupt, average_ending_capital_when_not_bankrupt = get_agent_statistics(dataframe, firm_ids_of_sophisticated_agents)
                print(f"\nThe sophisticated agents avoided bankruptcy in {percent_not_bankrupt:.2f} percent of the simulations.")
                if not math.isclose(percent_not_bankrupt, 0.0, rel_tol=0.0, abs_tol=1e-06):
                    print(f"When they avoided bankruptcy, they finished with an average capital of {average_ending_capital_when_not_bankrupt:.2f}.")
            
            # Analyze naive agent performance
            if firm_id_of_ai_agent_being_trained:
                percent_not_bankrupt, average_ending_capital_when_not_bankrupt = get_agent_statistics(dataframe, firm_ids_of_naive_agents)
                print(f"\nThe naive agents avoided bankruptcy in {percent_not_bankrupt:.2f} percent of the simulations.")
                if not math.isclose(percent_not_bankrupt, 0.0, rel_tol=0.0, abs_tol=1e-06):
                    print(f"When they avoided bankruptcy, they finished with an average capital of {average_ending_capital_when_not_bankrupt:.2f}.")
        print("------------------------------------------------")
            

if __name__ == "__main__":
    main()
