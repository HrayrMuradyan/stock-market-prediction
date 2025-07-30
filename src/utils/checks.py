def check_instance(name, val, expected_type):
    """
    Check if a value is an instance of the expected type.

    Parameters:
        name (str): Name of the variable (used in error messages).
        val (Any): The value to check.
        expected_type (type): The expected type of the value.

    Raises:
        TypeError: If `val` is not an instance of `expected_type`.
    """
    if not isinstance(val, expected_type):
        raise TypeError(f"Expected {name} to be a {expected_type.__name__}. Got {type(val).__name__}.")


def check_callable(name, val):
    """
    Check if a value is callable.

    Parameters:
        name (str): Name of the variable (used in error messages).
        val (Any): The value to check.

    Raises:
        TypeError: If `val` is not callable.
    """
    if not callable(val):
        raise TypeError(f"{name} must be callable.")


def check_subclass(name, val, base_class):
    """
    Check if a value is a subclass of a given base class.

    Parameters:
        val (Any): The value to check, expected to be a class type.
        base_class (type): The base class to compare against.
        name (str): Name of the variable (used in error messages).

    Raises:
        TypeError: If `val` is not a subclass of `base_class`.
    """
    if not issubclass(val, base_class):
        raise TypeError(f"{name} must be a subclass of {base_class.__name__}. Got {type(val).__name__}.")
    

def check_datatypes(schema_list):
    """
    Perform type and callable checks for a list of schema definitions.

    Each tuple in `schema_list` must follow one of these formats:
        - (name, val):        Checks that `val` is callable (via `check_callable`)
        - (name, val, type):  Determines whether to apply:
            - `check_subclass` if `val` is a class (i.e., a type), checking if it subclasses `type`
            - `check_instance` if `val` is an object instance, checking if it's an instance of `type`

    Parameters:
        schema_list (List[Tuple]): A list of validation rules.

    """

    # For each schema tuple
    for schema_tuple in schema_list:

        # If the len of the schema is 2
        if len(schema_tuple) == 2:

            # Check if it's callable
            name, val = schema_tuple
            check_callable(name, val)

        # If the len of the schema is 3
        elif len(schema_tuple) == 3:
            name, val, expected_type = schema_tuple

            # Check if both the val and expected type are class types
            if isinstance(val, type) and isinstance(expected_type, type):

                # Check if the val is a subclass of the expected type
                check_subclass(name, val, expected_type)

            # If not, then check the type (there is an instance mismatch)
            else:
                check_instance(name, val, expected_type)

        # Any other case is faulty
        else:
            raise ValueError(f"Invalid schema tuple: {schema_tuple}. Refer to the documentation for the correct setup.")

