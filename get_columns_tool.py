from langchain.tools import tool
import pandas as pd
@tool
def get_columns(path: str):
    """
    Read csv files and get the names of the columns.
    
    Parameters:
        path: str - Path to the CSV file.
    
    Returns:
        list: List of the column names.
    """
    
    print(pd.read_csv(path).columns)
    return list(pd.read_csv(path).columns)