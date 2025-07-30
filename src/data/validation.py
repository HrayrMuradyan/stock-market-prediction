import pandas as pd
import pandera.pandas as pa

def validate_data(merged_data):
    """
    Validates that all columns in the input DataFrame contain only 
    numeric values greater than 0 and have no missing values.

    Each column must:
    - Be of numeric type (int or float).
    - Contain only values strictly greater than 0.
    - Contain no null (NaN) values.

    Parameters
    ----------
    merged_data : pandas.DataFrame
        The input DataFrame to be validated.

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame if all checks pass.

    """
    # Input check
    if not isinstance(merged_data, pd.DataFrame):
        raise TypeError(f"Expected merged_data to be a dataframe. Got {type(merged_data).__name__}.")

    # Define the schema for checking the dataframe
    schema = pa.DataFrameSchema({

        # All columns of the dataframe
        col: pa.Column(
            
            # Should be an integer or a float
            float,

            # Be greater than 0
            checks=pa.Check.gt(0),

            # And should not be null
            nullable=False
        )
        for col in merged_data.columns  
    })

    # Validate the schema
    validated_data = schema.validate(merged_data)
    
    return validated_data