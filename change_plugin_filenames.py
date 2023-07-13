import os
import os.path as osp

OLD_NAME = 'eulerangles'
NEW_NAME = 'polarangles'
PLUGIN_DIR = './polarangles_plugin'

def change_files_name(dirname):
    file_names = [f.name for f in os.scandir(dirname) if f.is_file()]

    subdir_names = [p.name for p in os.scandir(dirname) if p.is_dir()]

    subdir_paths = [osp.join(dirname, x)
                    for x in subdir_names]

    for file_name in file_names:

        if file_name.find(OLD_NAME) > -1:
            new_file_name = file_name.replace(OLD_NAME,
                                              NEW_NAME)
            old_file_path = osp.join(dirname, file_name)
            new_file_path = osp.join(dirname, new_file_name)
            print(f'change {file_name} {new_file_name}')
            os.system(f'mv {old_file_path} {new_file_path}')

    for subdir in subdir_paths:
        change_files_name(subdir)


if __name__ == '__main__':

    change_files_name(PLUGIN_DIR)
