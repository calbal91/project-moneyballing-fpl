import numpy as np
import pandas as pd

def create_sql_table(dataframe, cursor, table_name, unique=True, verbose=True):
    
    '''
    Takes a dataframe and sqlite cursor and creates
    a table version of that dataframe
    in the sql database that the cursor is on
    '''
    #Get the column names (inc index)
    index_name = [dataframe.index.name]
    index_type = [dataframe.index.dtype]
    
    #If there's no specific name for the index, then give it a generic one
    if index_name == [None]:
        index_name = ['TableIndex']
        #Also assume that it's an integer primary key
        index_type = ['TEXT']
        
    columns = list(dataframe.columns)
    column_names = ", ".join(columns)
    columns = [f'{table_name}ID'] + index_name + columns
    
    #Get the data types of each
    column_types = [dataframe[i].dtype for i in dataframe.columns]
    
    #Transform the column types into SQL form
    type_mapper = {np.dtype('int64'): 'INTEGER',
                   np.dtype('float64'): 'REAL',
                   np.dtype('O'): 'TEXT'}
    
    column_types = [type_mapper[i] for i in column_types]
    column_types = ['INTEGER PRIMARY KEY'] + index_type + column_types
    
    #Zip the names and types together
    column_zip = list(zip(columns, column_types))
    
    cols_with_types = ""
    for i in column_zip:
        new_words = f'{i[0]} {i[1]}, '
        cols_with_types += new_words
    cols_with_types = cols_with_types[:-2]
    
    #If we have specified unique constraint, reflect this in the command
    if unique==True:
        sql_command = f"CREATE TABLE {table_name} ({cols_with_types}, unique ({column_names}));"
    else:
        sql_command = f"CREATE TABLE {table_name} ({cols_with_types});"
    
    if verbose==True:
        print(sql_command)
    
    #Include a try/except in case this is running for a second time...
    try:
        cursor.execute(sql_command)
        print('\nNew table added')
    except:
        print('\nTable not added - possibly already exists')
        
        
        
def populate_sql_from_dataframe(dataframe, sql_table, cursor, verbose=False):
    
    '''
    Takes a dataframe and the name of an sql table, and adds the dataframe
    rows to the bottom of the sql table associated with the cursor
    
    '''
     
    #Get the name of the index
    index_name = [dataframe.index.name]    
    #If there's no specific name for the index, then give it a generic one
    if index_name == [None]:
        index_name = ['TableIndex']

    #Isolate dataframe columns and create a single string containing them
    columns = index_name + list(dataframe.columns)
    column_string = ', '.join(columns)
        
    #Iterate through the rows in the dataframe
    for i in range(len(dataframe)):
        try:
            index = dataframe.index[i]
            row = dataframe.iloc[i]
            #Isolate the values in each row
            values = [index] + [dataframe.iloc[i][col] for col in dataframe.columns]
            #Put inverted commas around text objects
            values_strings = [f'"{i}"' if type(i)==str else f'{i}' for i in values]
            #Combine all values into single string
            values_string = ", ".join(values_strings)

            #Create the command...
            command = f'''INSERT INTO {sql_table} ({column_string})
                       VALUES ({values_string});
                       '''
            if verbose==True:
                print(command)
            #And excecute it...
            cursor.execute(command)
        except:
            print(f'Error on row {i} - you may be trying to upload a duplicate row')
        
        
def sql(query, cursor):
    '''
    Takes an SQL query string, and outputs a
    dataframe representation of the query result.
    '''
    #Execute the sql query
    cursor.execute(query)
    
    #Get the query into a dataframe and set columns
    df_temp = pd.DataFrame(cursor.fetchall())
    df_temp.columns = [x[0] for x in cursor.description]
    
    #Set the sql id as the dataframe index
    index_column = df_temp.columns[0]
    df_temp.set_index(index_column, drop=True, inplace=True)
    
    return df_temp