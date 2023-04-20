import pyodbc
import pandas as pd
import joblib
from keras.models import load_model
import re
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import os

# Define columns for the fact table
FACT_TABLE_COLUMNS = [
    "ProviderNetworkId",
    "ProviderId",
    "SpecialtyId",
    "LocationId",
    "Cost",
]


# class DatabaseConnection:
#     """
#     This class is a Singleton class that handles database connection to SQL Server.
#     """

#     __instance = None

#     @staticmethod
#     def get_instance():
#         """
#         Static access method for the Singleton class.
#         """
#         if DatabaseConnection.__instance is None:
#             DatabaseConnection()
#         # reistablish the connection if it has been closed
#         if DatabaseConnection.__instance.closed:
#             DatabaseConnection.__instance = pyodbc.connect(
#                 "Driver={SQL Server};"
#                 "Server=DESKTOP-J8TTBKS;"
#                 "[PotatoDataMartv1];"
#                 "Trusted_Connection=yes;"
#             )
        
#         return DatabaseConnection.__instance


class DatabaseConnection:
    """
    This class is a Singleton class that handles database connection to SQL Server.
    """

    __instance = None

    @staticmethod
    def get_instance(database_name="PotatoDataMartv1"):
        """
        Static access method for the Singleton class.
        """
        if DatabaseConnection.__instance is None:
            DatabaseConnection()
        # re-establish the connection if it has been closed
        if DatabaseConnection.__instance.closed:
            connection_string = f"Driver={{SQL Server}};Server=DESKTOP-J8TTBKS;Database={database_name};Trusted_Connection=yes;"
            DatabaseConnection.__instance = pyodbc.connect(connection_string)
        
        return DatabaseConnection.__instance


    def __init__(self):
        """
        Virtually private constructor for the Singleton class.
        """
        if DatabaseConnection.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            # Establish a connection to the database
            DatabaseConnection.__instance = pyodbc.connect(
                "Driver={SQL Server};"
                "Server=DESKTOP-J8TTBKS;"
                "[PotatoDataMartv1];"
                "Trusted_Connection=yes;"
            )


class Standardization:
    def __init__(self, columns):
        """
        Initialize the Standardization object with a list of columns to standardize.

        Parameters:
        columns (list): A list of column names to standardize.
        """
        self.columns = columns
        self.stemmer = SnowballStemmer("english")
        
    def __stem(self, text):
        """
        Apply stemming to the input text.

        Parameters:
        text (str): Input text to apply stemming on.

        Returns:
        str: Stemmed text.
        """
        # tokenize the text
        tokens = word_tokenize(text)
        # stem the tokens
        tokens = [self.stemmer.stem(token) for token in tokens]
        # join the tokens
        text = ' '.join(tokens)
        return text
    
    def __standarize_column(self, df, col):
        """
        Standardize the input column in the given dataframe.

        Parameters:
        df (pandas.DataFrame): Input dataframe.
        col (str): Name of the column to standardize.

        Returns:
        pandas.DataFrame: Dataframe with standardized column.
        """
        # convert the column to string type
        df[col] = df[col].astype(str)
        # lowercase the column
        df[col] = df[col].apply(lambda x: x.lower())
        # remove all punctuations
        df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        # remove all whitespaces
        df[col] = df[col].apply(lambda x: x.strip())
        # apply the stem function
        df[col] = df[col].apply(self.__stem)
        return df
    
    def standardize(self, df):
        """
        Standardize the specified columns in the input dataframe.

        Parameters:
        df (pandas.DataFrame): Input dataframe.

        Returns:
        pandas.DataFrame: Dataframe with standardized columns.
        """
        for col in self.columns:
            df = self.__standarize_column(df, col)    
        return df

    
class DatabaseHandler:
    def __init__(self, df, file_name):
        """
        Initializes a DatabaseHandler object.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data to be inserted into the database.
            file_name (str): The name of the file containing the data.
        """
        self.df = df
        self.db_connection = DatabaseConnection.get_instance()
        self.fact_table = pd.DataFrame(columns=FACT_TABLE_COLUMNS)
        self.provider_network_id = self._get_provider_network_id(file_name)
        self.__iterate_over_df()
    def __excecute_query(self, query, cursor):
        """
        Private method to execute the given query.

        Parameters:
            query (str): The query to be executed.
            cursor: The cursor object to interact with the database.
        """
        status = True
        error_message = ""
        try:
            cursor.execute(query)
        except pyodbc.Error as e:
            print(f"Error executing query: {query}")
            error_message = e.args[1]
            status = False
        
        return status,error_message, cursor
        
        
    def _get_provider_network_id(self, file_name):
        """
        Private method to get the ID of the provider network from the database or insert a new record if it doesn't exist.

        Parameters:
            file_name (str): The name of the file containing the data.

        Returns:
            int: The ID of the provider network.
        """
        cursor = self.db_connection.cursor()
        provider_network_code = file_name.split(".")[0]
        query = f"SELECT Id FROM PotatoDataMartv1.dbo.ProviderNetworkDim WHERE Code = '{provider_network_code}'"
        _,err,cursor = self.__excecute_query(query, cursor)
        row = cursor.fetchone()
        if row:
            cursor.close()
            return row[0]
        else:
            # Provider network not found, insert new record and get ID
            query = f"INSERT INTO PotatoDataMartv1.dbo.ProviderNetworkDim (Code, Description) VALUES ('{provider_network_code}', NULL)"
            status,err,cursor = self.__excecute_query(query, cursor)
            if status == False:
                return False,err
            self.db_connection.commit()
            # Get ID of inserted record
            query = f"SELECT Id FROM PotatoDataMartv1.dbo.ProviderNetworkDim WHERE Code = '{provider_network_code}'"
            _,err,cursor = self.__excecute_query(query, cursor)
            row = cursor.fetchone()
            cursor.close()
            return row[0]

    def _get_specialty_id(self, specialty_name, cursor):
        """
        Private method to get the ID of the specialty from the database or insert a new record if it doesn't exist.

        Parameters:
            specialty_name (str): The name of the specialty.
            cursor: The cursor object to interact with the database.

        Returns:
            int: The ID of the specialty.
        """
        # Check if specialty exists in database
        query = f"SELECT Id FROM PotatoDataMartv1.dbo.SpecialtyDim WHERE SpecialtyName = '{specialty_name}'"
        _,err,cursor = self.__excecute_query(query, cursor)
        row = cursor.fetchone()
        if row:
            specialty_id = row[0]
        else:
            # Specialty not found, insert new record and get ID
            query = f"INSERT INTO PotatoDataMartv1.dbo.SpecialtyDim (SpecialtyName) VALUES ('{specialty_name}')"
            status,err,cursor = self.__excecute_query(query, cursor)
            if status == False:
                return False,err
            self.db_connection.commit()
            # Get ID of inserted record
            query = f"SELECT Id FROM PotatoDataMartv1.dbo.SpecialtyDim WHERE SpecialtyName = '{specialty_name}'"
            _,err,cursor = self.__excecute_query(query, cursor)
            row = cursor.fetchone()
            specialty_id = row[0]
        return specialty_id

    def _get_city_id(self, city_name, cursor):
        """
        Private method to get the ID of the city from the database or insert a new record if it doesn't exist.

        Parameters:
            city_name (str): The name of the city.
            cursor: The cursor object to interact with the database.

        Returns:
            int: The ID of the city.
        """
        # Check if city exists in database
        query = f"SELECT Id FROM PotatoDataMartv1.dbo.LocationDim WHERE City = '{city_name}'"
        _,err,cursor = self.__excecute_query(query, cursor)

        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            # City not found, insert new record and get ID
            query = f"INSERT INTO PotatoDataMartv1.dbo.LocationDim (City) VALUES ('{city_name}')"
            status,err,cursor = self.__excecute_query(query, cursor)
            if status == False:
                return False,err
            self.db_connection.commit()
            # Get ID of inserted record
            query = f"SELECT Id FROM PotatoDataMartv1.dbo.LocationDim WHERE City = '{city_name}'"
            _,err,cursor = self.__excecute_query(query, cursor)
            row = cursor.fetchone()
            return row[0]

    def _insert_data(self, row, cursor):
        """
        Private method to insert a row of data into the database.

        Parameters:
            row (pandas.core.series.Series): The row of data to insert.
            cursor: The cursor object to interact with the database.
        """
        query = f"INSERT INTO PotatoDataMartv1.dbo.ProviderDim (ProviderName, startHour, endHour) VALUES ('{row['ProviderName']}', '{row['startHour']}', '{row['endHour']}')"
        status,err,cursor = self.__excecute_query(query, cursor)
        if status == False:
            return False,err
        self.db_connection.commit()
        query = f"SELECT Id FROM PotatoDataMartv1.dbo.ProviderDim WHERE ProviderName = '{row['ProviderName']}'"
        _,err,cursor = self.__excecute_query(query, cursor)
        returned_data = cursor.fetchone()
        provider_id = returned_data[0]
        specialty_id = self._get_specialty_id(row["SpecialtyName"], cursor)
        if isinstance(specialty_id,tuple) and specialty_id[0] == False:
            return False, specialty_id[1]
        city_id = self._get_city_id(row["City"], cursor)
        # check if city_id is a tuple
        if isinstance(city_id,tuple) and  city_id[0] == False:
            return False , city_id[1]
        query = f"INSERT INTO PotatoDataMartv1.dbo.CostFact (ProviderNetworkId, ProviderId, SpecialtyId, LocationId, Cost) VALUES ({self.provider_network_id}, {provider_id}, {specialty_id}, {city_id}, {row['Cost']})"
        status,err,cursor = self.__excecute_query(query, cursor)
        if status == False:
            return False,err
        self.db_connection.commit()
        return True

    def __iterate_over_df(self):
        """
        private method to Iterate over the data in the Pandas DataFrame and insert it into the database.
        """
        print("Inserting data into database...")

        cursor = self.db_connection.cursor()
        # for _, row in self.df.iterrows():
        #     print("Inserting data into database...")
        #     self._insert_data(row, cursor)
        self.df['info'] = self.df.apply(lambda row: self._insert_data(row, cursor), axis=1)
        # split the info into status and error
        self.df['status'] = self.df['info'].apply(lambda x: x[0] if isinstance(x,tuple) else x)
        self.df['error'] = self.df['info'].apply(lambda x: x[1] if isinstance(x,tuple) else "")
        # drop the info column
        self.df.drop('info', axis=1, inplace=True)
        
        cursor.close()
        # close the connection
        self.db_connection.close()
        


class FileReader:
    """
    This class handles reading the input file, transforming it into a DataFrame, and inserting the data into the database.
    """

    def __init__(self, filename, mapping_columns, combine_columns, seperator=","):
        """
        Constructor method for the FileReader class.

        Parameters:
            filename (str): The name of the file to be processed.
            mapping_columns (dict): A dictionary mapping input columns to output columns.
            combine_columns (dict): A dictionary specifying which columns to combine into a single column.
            seperator (str): The delimiter used in the input file.
        """
        self.filename = filename
        self.mapping_columns = mapping_columns
        self.combine_columns = combine_columns
        self.seperator = seperator
        self.standarizer = Standardization(['SpecialtyName', 'City'])

    def __get_df(self, file_name):
        """
        Private method to transform the input file into a DataFrame.

        Parameters:
            file_name (str): The name of the file to be processed.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """
        extention = file_name.split(".")[1]
        if extention == "txt" and self.seperator == ",":
            raise Exception(
                "no default separator for txt file you must pass separator as argument"
            )
        df = pd.read_csv(file_name, sep=self.seperator)
        for key, value in self.combine_columns.items():
            # convert the value columns into string type
            df[value] = df[value].astype(str)
            df[key] = df[value].apply(lambda x: " ".join(x), axis=1)
            df = df.drop(value, axis=1)

        df.rename(columns=self.mapping_columns, inplace=True)
        df["startHour"] = df["startHour"].astype("int8")
        df["endHour"] = df["endHour"].astype("int8")

        return df

    def __enter__(self):
        """
        Method to initialize the FileReader object.

        Returns:
            FileReader: The initialized FileReader object.
        """
        print("Transforming csv file into DataFrame")
        df = self.__get_df(self.filename)
        df = self.standarizer.standardize(df)
        db_handler = DatabaseHandler(df, self.filename)
        self.status = db_handler.df
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method to finalize the FileReader object.
        """
        print("Finished inserting data into database")




class Parser:
    def __init__(self):
        """
        Initialize the Parser object with the pre-trained model and tokenizers.

        """
        # Load the pre-trained model and tokenizers
        self.model = load_model('models/ner_model.h5')
        self.tokenizer = joblib.load('models/input_tokenizer.pkl')
        self.output_idx2wod = joblib.load('models/output_idx2word.pkl')

    def __clean_text(self, text):
        """
        Clean the input text by removing all non-alphanumeric characters and extra spaces.

        Parameters:
        text (str): Input text to be cleaned.

        Returns:
        str: Cleaned text.
        """
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def parse_text(self, text):
        """
        Parse the input text using the pre-trained model and return the predicted output.

        Parameters:
        text (str): Input text to be parsed.

        Returns:
        list: List of predicted output values.
        """
        # Clean the input text
        text = self.__clean_text(text)
        # Convert the text to sequences using the tokenizer
        text = self.tokenizer.texts_to_sequences([text])
        # Pad the sequences to a fixed length
        text = pad_sequences(text, maxlen=48, padding='post')
        # Predict the output using the pre-trained model
        pred = self.model.predict(text,verbose=0)
        # Convert the predicted output from indices to words
        pred = np.argmax(pred, axis=-1)
        pred = [self.output_idx2wod.get(idx) for idx in pred[0]]
        return pred


    


class FreeTextFileReader:
    """
    A class for parsing and standardizing free text files and inserting the relevant data into a database.
    """
    def __init__(self, filename):
        """
        Initialize the FreeTextFileReader object with the input filename.

        Parameters:
        filename (str): Name of the input file.
        """
        self.filename = filename
        self.parser = Parser()
        self.standarizer = Standardization(['SpecialtyName', 'City'])
        
    
    def clean_text(self, text):
        """
        Clean the input text by removing all non-alphanumeric characters and extra spaces.

        Parameters:
        text (str): Input text to be cleaned.

        Returns:
        str: Cleaned text.
        """
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def extract_text_by_label(self,row, label):
        """
        Extract the text for the given label from the input row.

        Parameters:
        row (pandas.Series): Input row containing text and NER tags.
        label (str): NER label to extract text for.

        Returns:
        str: Extracted text.
        """
        extracted_text = []
        tokens = row['text'].split(' ')
        ner = row['ner']
        for i in range(len(tokens)):
            if ner[i] == label:
                extracted_text.append(tokens[i])
            elif ner[i].startswith('I-') and ner[i][2:] == label[2:]:
                extracted_text.append(tokens[i])
            elif ner[i] == 'O':
                if len(extracted_text) > 0:
                    break
        if len(extracted_text) == 0:
            return None
        return ' '.join(extracted_text)
    
    def get_df(self):
        """
        Get the standardized dataframe from the input file.

        Returns:
        pandas.DataFrame: Standardized dataframe.
        """
        df = pd.read_csv(self.filename, sep='\n', header=None)
        # drop the first row
        df = df.drop(0, axis=0)
        # drop empty rows
        df = df.dropna()
        # rename the column to text
        df.columns = ['text']
        # call the clean_text function
        df['text'] = df['text'].apply(self.clean_text)
        # call the parse_text function
        df['ner'] = df['text'].apply(self.parser.parse_text)
        # extract the provider name
        df['ProviderName'] = df.apply(lambda x: self.extract_text_by_label(x,"B-Name"),axis=1)
        # extract the specialty
        df['SpecialtyName'] = df.apply(lambda x: self.extract_text_by_label(x,"B-Speciality"),axis=1)
        # extract the start hour
        df['startHour'] = df.apply(lambda x: self.extract_text_by_label(x,"B-Shour"),axis=1)
        # extract the end hour
        df['endHour'] = df.apply(lambda x: self.extract_text_by_label(x,"B-Ehour"),axis=1)
        # extract the cost
        df['Cost'] = df.apply(lambda x: self.extract_text_by_label(x,"B-Cost"),axis=1)
        # extract the city
        df['City'] = df.apply(lambda x: self.extract_text_by_label(x,"B-City"),axis=1)
        # drop the text and ner columns
        df = df.drop(['text','ner'],axis=1)
        # drop the rows with null values
        df = df.dropna()
        # reset the index
        df = df.reset_index(drop=True)
        return df
    
    def __enter__(self):
        """
        Method to initialize the FreeTextFileReader object.
        """
        # Get the standardized dataframe from the input file
        df = self.get_df()
        
        # Standardize the relevant columns in the dataframe
        df = self.standarizer.standardize(df)
        # Initialize the DatabaseHandler object with the dataframe and input filename
        db_handler = DatabaseHandler(df, self.filename)
        self.status = db_handler.df
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method to finalize the FileReader object.
        """
        print("Finished inserting data into database")
        
        

class OLTPReader:

    def __init__(self, database_name):
        self.db_connection = DatabaseConnection.get_instance(database_name)
        self.cursor = self.db_connection.cursor()

    


    
    def __read_from_db(self,cursor):
        QUERY = """
        SELECT Potato.dbo.ProviderNetwork.Code as ProviderNetwork,
        Potato.dbo.Provider.FirstName,
        Potato.dbo.Provider.LastName,
        Potato.dbo.Specialty.Name As Speciality,
        Potato.dbo.City.Name as City,
        MAX(CASE WHEN Potato.dbo.attribute.AttributeName = 'Cost' THEN Potato.dbo.ProviderAttribute.AttributeValue END) AS Cost,
        MAX(CASE WHEN Potato.dbo.attribute.AttributeName = 'W_HOURS_START' THEN Potato.dbo.ProviderAttribute.AttributeValue END) AS W_HOURS_START,
        MAX(CASE WHEN Potato.dbo.attribute.AttributeName = 'W_HOURS_END' THEN Potato.dbo.ProviderAttribute.AttributeValue END) AS W_HOURS_END
        FROM Potato.dbo.Provider
        JOIN Potato.dbo.ProviderSpecialtyAssoc ON Potato.dbo.Provider.Id = Potato.dbo.ProviderSpecialtyAssoc.ProviderId 
        JOIN Potato.dbo.Specialty ON Potato.dbo.ProviderSpecialtyAssoc.SpacialtyId = Potato.dbo.Specialty.Id
        JOIN Potato.dbo.ProviderNetwork ON Potato.dbo.Provider.ProviderNetworkId = Potato.dbo.ProviderNetwork.Id
        JOIN Potato.dbo.ProviderAttribute ON Potato.dbo.Provider.Id = Potato.dbo.ProviderAttribute.ProviderId
        JOIN Potato.dbo.attribute ON Potato.dbo.ProviderAttribute.AttributeId = Potato.dbo.attribute.Id
        Join Potato.dbo.ProviderAddress on Potato.dbo.ProviderAddress.ProviderId = Potato.dbo.Provider.Id
        Join Potato.dbo.City on Potato.dbo.ProviderAddress.CityId = Potato.dbo.City.Id

        WHERE Potato.dbo.attribute.AttributeName IN ('W_HOURS_START', 'W_HOURS_END', 'Cost')
        GROUP BY Potato.dbo.ProviderNetwork.Code, Potato.dbo.Provider.Id, Potato.dbo.Provider.FirstName, Potato.dbo.Provider.LastName, Potato.dbo.Specialty.Name,Potato.dbo.City.Name;
        """
        cursor.execute(QUERY)
        rows = cursor.fetchall()
        columns = ["ProviderNetwork","Fname","Lname","Specialty","City","Cost","Start Hour","End Hour"]
        # convert rows to tuples
        tuples = [tuple(row) for row in rows]
        df = pd.DataFrame(tuples, columns=columns)
        return df
    def __get_dfs(self):
        df = self.__read_from_db(self.cursor)
        # split the df to list of df's based on provider network
        df_list = [df[df['ProviderNetwork'] == provider_network] for provider_network in df['ProviderNetwork'].unique()]
        # create a list of tuples with the provider network and the df and drop the provider network column
        df_list = [(provider_network, df.drop(columns=['ProviderNetwork'])) for provider_network, df in zip(df['ProviderNetwork'].unique(), df_list)]
        return df_list
    def __iter_through_dfs(self,df_list):
        vendor_columns_mapping = {
        "Specialty" : "SpecialtyName",
        "Start Hour" : "startHour",
        "End Hour" : "endHour"
        }
        combined_columns_mapping = {
            "ProviderName" : ["Fname","Lname"],
        }
        self.dfs = []
        # save the df's to csv files temporarily
        for provider_network, df in df_list:
            # save the df to csv file named after the provider network
            df.to_csv(f"data/temp/{provider_network}.csv", index=False)
            # insert the data using FileReader
            
            with FileReader(f"data/temp/{provider_network}.csv", vendor_columns_mapping, combined_columns_mapping) as reader:
                self.dfs.append(((provider_network),reader.status))
            # delete the csv file
            os.remove(f"data/temp/{provider_network}.csv")
            
    def __enter__(self):
        df_list = self.__get_dfs()
        self.__iter_through_dfs(df_list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Finished inserting data into database")