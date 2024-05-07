import pandas as pd

################################################################################
# The instance data is stored in the data\instances folder
################################################################################

def read_instance(instance_name: str):
    '''
    Reads a CSV file based on the instance_name provided, 
    skips the header, sets the separator to one or more whitespaces
    
    Parameters:
    instance_name (str): Name of the instance file to be read without extension
    
    Returns:
    (DataFrame): Dataframe with the instance data
    '''
    
    return pd.read_csv(f"data\instances\{instance_name}.vrp", 
                       header=None, sep='\s+', 
                       names=[f'{i}' for i in range(8)]).iloc[:,0:5]
    
def obtain_instance_data(instance_name: str):
    ''' 
    Extracts the points, demands, number of vehicles, number of nodes, 
    and capacity of each vehicle from the instance data. It packages 
    these into a dictionary and returns this dictionary.
    
    Patameters:
    instance_name (str): Name of the instance file to be read without extension
    
    Returns:
    data (dict): Dictionary with the instance data
    '''
    
    instance = read_instance(instance_name)
    
    points = instance.iloc[2:,1:3].reset_index(drop=True)
    demands = instance.iloc[2:,-1].reset_index(drop=True)
    
    n_vehicles = instance.iloc[0,1]
    n_nodes = instance.iloc[0,2]
    q_vehicles = instance.iloc[1,1]
    
    data = {'points':points, 'demands': demands, 
            'n_vehicles':n_vehicles, 'n_nodes': n_nodes, 
            'q_vehicles': q_vehicles}
    
    return data