import os.path
import re
from batoolset.files.tools import get_home_dir

def get_JDK_HOME(java_version=8):
    return _get_Java_HOME(jdk_or_jre='.jdk',java_version=java_version)

def get_JRE_HOME(java_version=8):
    return _get_Java_HOME(jdk_or_jre='.jre',java_version=java_version)
def _get_Java_HOME(jdk_or_jre='.jdk',java_version=8):
    home_dir = get_home_dir(jdk_or_jre)
    if not os.path.exists(home_dir):
        return None
    folder_path=os.path.join(home_dir,'jdk'+str(java_version))
    if not os.path.exists(folder_path):
        print(('failed'))
        return None
    # List all directories in the folder
    directories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    if not directories:
        return

    # print('directories1',directories)

    if len(directories)==1:
        return os.path.join(folder_path,directories[0])
    else:
        return os.path.join(folder_path,find_most_likely_java_folder(directories, java_version=java_version))

    # print(directories)



# Function to find folders containing "java8"
def find_most_likely_java_folder(folder_list, java_version=8):
    # Filter folders that contain the specified Java version and 'jdk'
    java_folders = [folder for folder in folder_list if str(java_version) in folder]

    # print('java_folders', java_folders)
    if not java_folders:
        return None

    # print('java_folders2',java_folders)

    # Sort folders by version number
    java_folders.sort(
        key=lambda x: int(re.search(rf'{java_version}u(\d+)', x).group(1)) if re.search(rf'{java_version}u(\d+)', x) else float('inf'))

    return java_folders[0]


if __name__ == '__main__':

    if False:
        test = ['jdk8u422-b05', 'jdk-11.0.24+8']
        print(find_most_likely_java_folder(test,11))

    print(get_JDK_HOME(8))
    print(get_JRE_HOME(8))
    print(get_JDK_HOME(11))
