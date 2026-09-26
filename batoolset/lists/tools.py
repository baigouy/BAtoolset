# contains a set of useful methods for list and list operations
import itertools
import itertools
import sys
from itertools import combinations

def compare_lists_sets(list1, list2):
    return set(list1) == set(list2)

def divide_list_into_sublists(lst, num_rows, num_cols):
    # Calculate the total number of elements
    total_elements = num_rows * num_cols

    # If the list has fewer elements than expected, add None to fill the gaps
    if len(lst) < total_elements:
        lst.extend([None] * (total_elements - len(lst)))

    # Divide the list into sublists
    sublists = [lst[i:i + num_cols] for i in range(0, total_elements, num_cols)]

    return sublists

def bool_list_to_index_list(bool_list):
    index_list = []
    for i, val in enumerate(bool_list):
        if val:
            index_list.append(i)
    return index_list

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def combine_lists(list1, list2):
    combined_list = [(item1, item2) for item1 in list1 for item2 in list2]
    return combined_list

def flatten_iterable(iterable):
    non_iterable_objects = []

    for obj in iterable:
        # If the object is not iterable, add it to the list of non-iterable objects
        if not is_iterable(obj):
            non_iterable_objects.append(obj)
        # If the object is iterable, recursively call the function on the object
        else:
            non_iterable_objects.extend(flatten_iterable(obj))

    return non_iterable_objects


def find_first_object_of_type(obj_list, obj_type):
    try:
        # Find the first object of the given type in the list
        obj = next(obj for obj in obj_list if isinstance(obj, obj_type))
        return obj
    except StopIteration:
        # No object of the given type was found in the list
        return None


def find_all_objects_of_type(obj_list, obj_type):
    if not obj_list:
        return None
    # Find all the objects of the given type in the list
    objects = [obj for obj in obj_list if isinstance(obj, obj_type)]
    return objects

def find_first_index(element, my_list):
    try:
        # Find the index of the first occurrence of the element in the list
        index = my_list.index(element)
        return index
    except ValueError:
        # The element is not in the list
        return None

def get_list_combinatorial(lst, repack_all_to_lists_if_not_lists=True, return_list=True):
    """
    Creates all possible combinations of the elements in a list.

    Args:
        lst (list): The input list.
        repack_all_to_lists_if_not_lists (bool): If True, converts non-list elements to lists.
        return_list (bool): If True, returns the combinations as a list of lists.

    Returns:
        list: The list of all possible combinations.

    Examples:
        >>> lst = ['D', ['1P', 'hinge_undefined'], '1P']
        >>> result = get_list_combinatorial(lst)
        >>> print(result)
        [['D', '1P', '1P'], ['D', 'hinge_undefined', '1P']]
    """
    optimized_list = lst

    if repack_all_to_lists_if_not_lists:
        optimized_list = [[item] if not isinstance(item, list) else item for item in lst]

    tmp = list(itertools.product(*optimized_list))

    if return_list:
        tmp = [list(el) for el in tmp]

    return tmp


def list_contains_sublist(lst):
    """
    Checks if a list contains any sublists.

    Args:
        lst (list): The input list.

    Returns:
        bool: True if the list contains sublists, False otherwise.

    Examples:
        >>> lst = ['a', 'b', ['c', 'd']]
        >>> result = list_contains_sublist(lst)
        >>> print(result)
        True
    """
    for el in lst:
        if isinstance(el, list):
            return True

    return False

def get_consecutive_file_pairs(file_paths):
    """
    Given a list of file paths, returns a list of tuples,
    where each tuple contains the consecutive pair of file paths.

    Args:
        file_paths (list): The list of file paths.

    Returns:
        list: The list of tuples containing consecutive file pairs.

    Examples:
        >>> file_paths = ['file1.txt', 'file2.txt', 'file3.txt']
        >>> result = get_consecutive_file_pairs(file_paths)
        >>> print(result)
        [('file1.txt', 'file2.txt'), ('file2.txt', 'file3.txt')]
    """
    return [(file_paths[i], file_paths[i+1]) for i in range(len(file_paths)-1)]

def flatten_list(input_list):
    """
    Flattens a list containing nested tuples or other elements.

    Args:
        input_list (list): The input list to be flattened.

    Returns:
        list: A new list with nested elements flattened.

    Examples:
        >>> input_list = ['toto', 'tutu', ('bob', 'beb'), 'tata']
        >>> flattened_list = flatten_list(input_list)
        >>> print(flattened_list)
        ['toto', 'tutu', 'bob', 'beb', 'tata']
    """
    # Flatten the list using a list comprehension
    flattened_list = [item for sublist in input_list for item in (sublist if isinstance(sublist, (tuple, list)) else [sublist])]
    return flattened_list

def create_all_possible_pairs(single_cutters):
    """
    Create all possible pairs of keys from a dictionary of single-cut restriction enzymes.

    Args:
        single_cutters (dict): A dictionary where keys represent enzyme names and values
                              are associated information for single-cut restriction enzymes.

    Returns:
        list: A list of tuples containing all unique pairs of enzyme names from the dictionary.

    Examples:
        >>> single_cutters = {'EcoRI': {'site': 'GAATTC', 'recognition_site': 'G^AATTC', 'cut_site': 1},
        ...                   'BamHI': {'site': 'GGATCC', 'recognition_site': 'G^GATCC', 'cut_site': 1}}
        >>> pairs = create_all_possible_pairs(single_cutters)
        >>> print(pairs)
        [('EcoRI', 'BamHI')]
        >>> single_cutters = ['EcoRI','BamHI']
        >>> pairs = create_all_possible_pairs(single_cutters)
        >>> print(pairs)
        [('EcoRI', 'BamHI')]
    """
    if isinstance(single_cutters, dict):
        single_cutters =single_cutters.keys()
    key_pairs = [pair for pair in combinations(single_cutters, 2)]
    return key_pairs

def swap_items(lst, index1, index2):
    # Check if the indices are valid
    if index1 < 0 or index1 >= len(lst) or index2 < 0 or index2 >= len(lst):
        print("Invalid indices")
        return

    print('before swapping', lst)
    # Swap the items
    temp = lst[index1]
    lst[index1] = lst[index2]
    lst[index2] = temp

    print("Swapped items: ", lst)


# def move_left(lst, indices, distance):
#     # Create a copy of the list
#     lst = lst[:]
#
#     # Sort the indices in descending order
#     indices = sorted(indices, reverse=True)
#
#     # Move each element to the left by the given distance
#     for index in indices:
#         element = lst.pop(index)
#         lst.insert(index - distance, element)
#
#     return lst

# def move_left(lst, indices, distance):
#     # Create a copy of the list
#     lst = lst[:]
#
#     # Sort the indices in descending order
#     indices = sorted(indices, reverse=True)
#
#     # Move each element to the left by the given distance
#     for index in indices:
#         element = lst.pop(index)
#         lst.insert((index - distance) % len(lst), element)
#
#     return lst
# def move_left(lst, indices, distance):
#     """
#     Moves the elements at the specified indices to the left by a given distance in the list.
#     Returns a new list with the elements moved.
#     """
#     new_lst = lst[:]  # Make a copy of the original list
#     indices = sorted(indices, reverse=True)  # Sort indices in descending order
#
#     for idx in indices:
#         if idx < distance:
#             continue  # Skip if the index is smaller than the distance
#
#         elem = new_lst.pop(idx)  # Remove the element at the index
#         new_lst.insert(idx - distance, elem)  # Insert the element at the new position
#
#     return new_lst
def move_left(lst, indices, distance=1):
    """
    Moves the elements at the specified indices to the left by a given distance in the list.
    Returns a new list with the elements moved.
    """
    new_lst = lst[:]  # Make a copy of the original list
    indices = sorted(indices)  # Sort indices in ascending order

    for idx in indices:
        if idx < distance:
            new_lst.insert(0, new_lst.pop(idx))  # Move element to the front
        else:
            elem = new_lst.pop(idx)
            new_lst.insert(idx - distance, elem)

    return new_lst

def move_right(lst, indices, distance=1):
    """
    Moves the elements at the specified indices to the right by a given distance in the list.
    Returns a new list with the elements moved.
    """
    new_lst = lst[:]  # Make a copy of the original list
    indices = sorted(indices, reverse=True)  # Sort indices in descending order

    for idx in indices:
        if idx + distance >= len(new_lst):
            elem = new_lst.pop(idx)
            new_lst.append(elem)  # Move element to the end
        else:
            elem = new_lst.pop(idx)
            new_lst.insert(idx + distance, elem)

    return new_lst


# that is the best I have so far
# def move_up(items, selected_indices):
#     # Sort selected indices in descending order to prevent interference with previous movements
#     selected_indices.sort(reverse=True)
#     for index in selected_indices:
#         if index > 0:
#             # Swap the item at the current index with the item above it
#             items[index], items[index - 1] = items[index - 1], items[index]
#
# def move_down(items, selected_indices):
#     # Sort selected indices in ascending order to prevent interference with previous movements
#     selected_indices.sort()
#     for index in selected_indices:
#         if index < len(items) - 1:
#             # Swap the item at the current index with the item below it
#             items[index], items[index + 1] = items[index + 1], items[index]


# def move_right(lst, indices, distance):
#     """
#     Moves the elements at the specified indices to the right by a given distance in the list.
#     Returns a new list with the elements moved.
#     """
#     new_lst = lst[:]  # Make a copy of the original list
#     indices = sorted(indices, reverse=True)  # Sort indices in descending order
#
#     # Keep track of the new index of each element as it is moved
#     new_indices = []
#
#     for idx in indices:
#         if new_indices:
#             # Adjust the distance for subsequent elements based on the new index of the previous element
#             last_new_idx = new_indices[-1]
#             if last_new_idx + 1 < len(new_lst):
#                 distance = min(distance, last_new_idx + 1 - idx)
#             else:
#                 distance = 0
#
#         if idx + distance >= len(new_lst):
#             elem = new_lst.pop(idx)
#             new_lst.append(elem)  # Move element to the end
#             new_indices.append(len(new_lst) - 1)
#         else:
#             elem = new_lst.pop(idx)
#             new_lst.insert(idx + distance, elem)
#             new_indices.append(idx + distance)
#
#     return new_lst



# def move_left(lst, indices, distance):
#     """
#     Moves the elements at the specified indices to the left by a given distance in the list.
#     Preserves the original order of the elements being moved.
#     Returns a new list with the elements moved.
#     """
#     new_lst = lst[:]  # Make a copy of the original list
#     indices = sorted(indices)  # Sort indices in ascending order
#
#     moved_elems = []  # Store the elements being moved
#     for idx in indices:
#         elem = new_lst.pop(idx)
#         if idx < len(lst) - distance:
#             moved_elems.append(elem)
#         else:
#             new_lst.insert(idx - distance, elem)
#
#     # Insert the moved elements at the beginning of the list
#     new_lst[:0] = moved_elems
#
#     return new_lst
# def move_items_by_distance(data, indices, distance, direction="up"):
#     temp_list = [data[i] for i in indices]
#     for i in sorted(indices, reverse=True):
#         del data[i]
#
#     if direction == "up":
#         target_index = min(indices) - distance
#     else:  # direction == "down"
#         target_index = max(indices) + distance + 1
#
#     # Ensure target_index is within list bounds
#     target_index = max(0, min(target_index, len(data)))
#
#     data[target_index:target_index] = temp_list

if __name__ == '__main__':


    if True:
        # Define a list
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Define the number of rows and columns
        num_rows = 3
        num_cols = 3

        # Divide the list into sublists
        sublists = divide_list_into_sublists(lst, num_rows, num_cols)

        # Print the sublists
        for sublist in sublists:
            print(sublist)

        sys.exit(0)

    if True:
        # Example usage
        lst = [0, 1, 2, 3, 4, 5, 6, 7]
        indices = [1, 6]
        distance = 5
        print(move_left(lst, indices, distance))  # Output: [1, 0, 2, 3, 4, 6, 5, 7]

        lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        indices = [1, 5, 6]
        distance = 10

        new_lst = move_right(lst, indices, distance)
        print(new_lst)  # Output: [0, 2, 3, 1, 4, 5, 7, 6]

        # # Example usage:
        # my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8] #['a', 'b', 'c', 'd', 'e', 'f']
        # indices_to_move = [1, 3, 5]  # Move 'b', 'd', and 'f'
        #
        # move_items_by_distance(my_list, indices_to_move,1, direction="up")
        # print(my_list)  # Output: ['a', 'c', 'e', 'b', 'd', 'f']

        sys.exit(0)

    if True:
        import os

        # list of file paths as strings
        file_paths = ["/path/to/file1.txt", "/path/to/file2.txt", "/path/to/file3.txt"]

        print(get_consecutive_file_pairs(file_paths))

    test = ['D', ['1P', 'hinge_undefined'], '1P']
    print(list_contains_sublist(test))  # True
    combi = get_list_combinatorial(test)
    print(combi)  # [['D', '1P', '1P'], ['D', 'hinge_undefined', '1P']]
    print(list_contains_sublist(combi[0]))  # False
