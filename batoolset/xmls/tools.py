import xml.etree.ElementTree as ET

from batoolset.files.tools import get_consolidated_filename_from_parent


def _replace_filename_tags(file_path):
    # just a small trick to allow template mode in PyFigures
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the <filename> tags and replace their content with '@template@'
    for elem in root.iter('filename'):
        if elem.text and elem.text !='None':
            elem.text = '""@template@""'

    # Overwrite the original file with the modified XML
    tree.write(file_path)


# def _remove_insets(file_path):
#     # just a small trick to allow template mode in PyFigures
#     # Parse the XML file
#     tree = ET.parse(file_path)
#     root = tree.getroot()
#
#     # Find the <annotations> tag
#     annotations_tag = root.find('annotations')
#     print('annotations_tag', annotations_tag)
#     if annotations_tag is not None:
#         # Iterate over its children and remove the <image2D> tags
#         for elem in annotations_tag.iter('Image2D'):
#             print('removing', elem)
#             annotations_tag.remove(elem)
#
#     # Overwrite the original file with the modified XML
#     tree.write(file_path)




def _update_filename_of_consolidated_files(file_path, xml_string):
    # just a small trick to allow template mode in PyFigures
    # Parse the XML file
    # tree = ET.parse(file_path)
    root = ET.fromstring(xml_string)
    # root = tree.getroot()

    # Find the <filename> tags and replace their content with '@template@'
    for elem in root.iter('filename'):
        # elem.text = '""@template@""'
        if elem.text == 'None':
            continue

        value = elem.text
        if isinstance(value, str) and value.startswith('""') and value.endswith('""'):
            value = value[2:-2]
            elem.text = f'""{get_consolidated_filename_from_parent(value,file_path)}""'

    # Overwrite the original file with the modified XML
    # tree.write(file_path)
    # Convert the modified XML back to a string
    xml_string = ET.tostring(root, encoding='unicode')  # encoding='unicode' returns a string, otherwise it returns a bytes object

    return xml_string
