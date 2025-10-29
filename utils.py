import pandas as pd
import pandera as pa

def load_and_validate_csv(filepath, schema):
    df = pd.read_csv(filepath)
    schema.validate(df, lazy=True)
    return df
