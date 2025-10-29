import pytest
import pandas as pd
import pandera.pandas as pa
from dataset_schema import historical_prices_schema, all_news_schema, historical_prices_impact_schema, trade_log_schema, trade_log_summary_schema, aggregated_news_schema, vectorized_news_schema
from utils import load_and_validate_csv
import os

# Define the directory where your datasets are stored
DATASET_DIR = "datasets"

def get_dataset_files():
    datasets = {
        "historical_prices.csv": historical_prices_schema,
        "all_news.csv": all_news_schema,
        "historical_prices_impact.csv": historical_prices_impact_schema,
        "aggregated_news.csv": aggregated_news_schema, # Now uses its own schema
        "trade_log.csv": trade_log_schema, 
        "final_summary.csv": trade_log_summary_schema, 
        "vectorized_news_dtm.csv": vectorized_news_schema, # New schema
        "vectorized_news_tfidf.csv": vectorized_news_schema, # New schema
        "vectorized_news_curated.csv": vectorized_news_schema, # New schema
        "sample_historical_prices.csv": historical_prices_schema,
        "sample_all_news.csv": all_news_schema,
        "sample_historical_prices_impact.csv": historical_prices_impact_schema,
        "sample_aggregated_news.csv": aggregated_news_schema, 
        "sample_trade_log.csv": trade_log_schema, 
        "sample_final_summary.csv": trade_log_summary_schema, 
        "sample_vectorized_news_dtm.csv": vectorized_news_schema, 
        "sample_vectorized_news_tfidf.csv": vectorized_news_schema, 
        "sample_vectorized_news_curated.csv": vectorized_news_schema, 
    }
    
    # Removed filtering logic to include all datasets defined, even if files don't exist.
    # The test function will now explicitly handle FileNotFoundError.
    if not datasets:
        pytest.skip("No dataset definitions found for testing.")

    return datasets


@pytest.mark.parametrize("filename, schema", get_dataset_files().items())
def test_dataset_schema_validation(filename, schema):
    filepath = os.path.join(DATASET_DIR, filename)
    print(f"Validating {filepath} against schema: {schema}")

    try:
        df = load_and_validate_csv(filepath, schema)
    except FileNotFoundError:
        pytest.fail(f"File not found: {filepath}")

    try:
        schema.validate(df, lazy=True)
        print(f"Validation successful for {filepath}")
    except pa.errors.SchemaErrors as err:
        error_message = f"Schema validation failed for {filepath}:\n"
        for failure_case in err.failure_cases.itertuples():
            error_message += (
                f"  Column: {failure_case.column}, "
                f"Row: {failure_case.index}, "
                f"Value: {failure_case.failure_case}, "
                f"Check: {failure_case.check}" + "\n"
            )
        pytest.fail(error_message)