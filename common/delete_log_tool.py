
import numpy as np
import os.path as osp
import os
import shutil
import glob
import stat

from common.private_config import *


class DeleteLogTool(object):
    def __init__(self, log_root, sub_proj, task, regex, optional_log_type=None):
        self.log_root = log_root
        self.sub_proj = sub_proj
        self.task = task
        self.regex = regex
        self.log_types = LOG_TYPE.copy()
        if optional_log_type is not None:
            self.log_types.extend(optional_log_type)

    def delete_related_log(self):
        for log_type in self.log_types:
            root_dir_regex = osp.join(self.log_root, self.sub_proj, log_type, self.task, self.regex)
            empty = True
            for root_dir in glob.glob(root_dir_regex):
                empty = False
                if os.path.exists(root_dir):
                    for file_list in os.walk(root_dir):
                        for name in file_list[2]:
                            try:
                                os.remove(os.path.join(file_list[0], name))
                            except Exception as e:
                                import traceback
                                err_str = traceback.format_exc()
                                print(err_str)
                                print("try \'\'sudo chmod -Rc 777 ./your_path/\'\' to solve the permission problem")
                            # print("delete file {}".format(name))
                    if os.path.isdir(root_dir):
                        try:
                            shutil.rmtree(root_dir)
                        except Exception as e:
                            import traceback
                            err_str = traceback.format_exc()
                            print(err_str)
                            print("try \'\'sudo chmod -Rc 777 ./your_path/\'\' to solve the permission problem")
                        print("delete dir {}".format(root_dir))
                    else:
                        os.remove(root_dir)
                        print("delete file {}".format(root_dir))
                else:
                    print("not dir {}".format(root_dir))
            if empty: print("empty regex {}".format(root_dir_regex))
if __name__ == '__main__':
    dlt = DeleteLogTool("../", "vae", "ae", "2019/09/29/17*")
    dlt.delete_related_log()