import json

def serialize_dict_to_file(data, file_path):
    """
    Serialize a nested dictionary to a JSON file.

    Parameters:
    - data (dict): The dictionary to serialize.
    - file_path (str): The path to the file where the dictionary will be saved.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)  # indent=4 for pretty-printing

def deserialize_dict_from_file(file_path):
    """
    Deserialize a nested dictionary from a JSON file.

    Parameters:
    - file_path (str): The path to the file from which the dictionary will be loaded.

    Returns:
    - dict: The deserialized dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

if __name__ == '__main__':
    nested_dict = {
        'key1': 'value1',
        'key2': {
            'subkey1': 'subvalue1',
            'subkey2': {
                'subsubkey1': 'subsubvalue1',
                'subsubkey2': 'subsubvalue2'
            }
        },
        'key3': ['list_item1', 'list_item2', 'list_item3']
    }

    # Serialize the nested dictionary to a JSON file
    serialize_dict_to_file(nested_dict, 'nested_dict.json')

    # Deserialize the nested dictionary from the JSON file
    loaded_dict = deserialize_dict_from_file('nested_dict.json')

    print(loaded_dict)