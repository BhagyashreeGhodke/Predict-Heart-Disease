import os
import mysql.connector
from dotenv import load_dotenv

def initialize_database():
    """
    Connects to the database and creates the 'predictions' table if it doesn't exist.
    This function is called automatically when the Flask app starts.
    """
    # Load environment variables from .env file, searching up the directory tree
    # This ensures it finds the .env file in the project root
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    print("Checking database and tables...")

    MYSQL_HOST = os.getenv('MYSQL_HOST')
    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')
    MYSQL_PORT = os.getenv('MYSQL_PORT')

    if not all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE]):
        print("Error: Database credentials not fully set in .env file. Skipping DB initialization.")
        return

    conn = None
    try:
        # Establish connection
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            port=int(MYSQL_PORT)
        )
        cursor = conn.cursor()
        
        table_name = 'predictions'
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        if cursor.fetchone():
            print(f"Table '{table_name}' already exists.")
        else:
            create_table_query = """
            CREATE TABLE predictions (
                id INT AUTO_INCREMENT PRIMARY KEY, HighBP FLOAT, HighChol FLOAT, CholCheck FLOAT,
                BMI FLOAT, Smoker FLOAT, Stroke FLOAT, Diabetes FLOAT, PhysActivity FLOAT,
                Fruits FLOAT, Veggies FLOAT, HvyAlcoholConsump FLOAT, AnyHealthcare FLOAT,
                NoDocbcCost FLOAT, GenHlth FLOAT, MentHlth FLOAT, PhysHlth FLOAT,
                DiffWalk FLOAT, Sex FLOAT, Age FLOAT, Education FLOAT, Income FLOAT,
                predictionResult VARCHAR(255), predictionConfidence VARCHAR(255),
                createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
            print(f"Table '{table_name}' not found. Creating it now...")
            cursor.execute(create_table_query)
            print(f"Table '{table_name}' created successfully.")
        
        cursor.close()

    except mysql.connector.Error as err:
        print(f"Database setup error: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()

def get_db_connection():
    """Establishes a connection to the MySQL database for operations."""
    # .env should already be loaded by initialize_database() at startup
    conn = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE'),
        port=int(os.getenv('MYSQL_PORT'))
    )
    return conn

def save_prediction_to_db(input_data, prediction_text, confidence_score_formatted):
    """Saves the user's input data and the model's prediction to the database."""
    conn = None
    try:
        conn = get_db_connection()
        data_to_save = input_data.copy()
        data_to_save['predictionResult'] = prediction_text
        # The formatted string (e.g., "95.45%") is now saved directly
        data_to_save['predictionConfidence'] = confidence_score_formatted

        # --- ADDED PRINT STATEMENT ---
        print("\n--- Saving to Database ---")
        for key, value in data_to_save.items():
            print(f"  {key}: {value}")
        print("--------------------------\n")
        # -----------------------------

        cols = ", ".join([f"`{k}`" for k in data_to_save.keys()])
        placeholders = ", ".join(["%s"] * len(data_to_save))
        values = tuple(data_to_save.values())
        query = f"INSERT INTO predictions ({cols}) VALUES ({placeholders})"

        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        print("Successfully saved prediction to the database.")
        return True
    except mysql.connector.Error as err:
        print(f"Error saving to DB: {err}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()
