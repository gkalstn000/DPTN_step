import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def get_csv_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.csv'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list'%flist)
    return []
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'lightcoral', 'orangered']
csv_list = get_csv_list('./eval_results')
csv_list = ['./eval_results/DPTN.csv', './eval_results/c_to_c.csv', './eval_results/c_to_s.csv', './eval_results/s_to_s.csv', './eval_results/DPTN_2_stage.csv']
df_dict = {}
for file_path in csv_list:
    file_name = file_path.split('/')[-1].replace('.csv', '')
    df_dict[file_name] = pd.read_csv(file_path)

barWidth = 0.1
fig = plt.subplots(figsize =(12, 8))

for idx, (name, df) in enumerate(df_dict.items()) :
    courses = [x + barWidth * idx for x in range(12)]
    values = list(df_dict[name].iloc[2][2:])
    plt.bar(courses, values, color=colors[idx], width=barWidth, edgecolor='grey', label=name)


plt.xlabel("# of keypoint")
plt.ylabel(f"{name} score")
plt.xticks([r + barWidth for r in range(len(df_dict[name].columns[2:]))],
        [str(i) for i in range(7, 19)])
plt.legend()
plt.show()