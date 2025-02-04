
import os
import glob

PATHs = ['/home/acszy/2025/Affirmative/data/select_10_result_*.csv','/home/acszy/2025/Affirmative/data/select_1_*.csv']
for path in PATHs:
    for file_path in glob.glob(path):
        if os.path.exists(file_path):
            os.remove(file_path)