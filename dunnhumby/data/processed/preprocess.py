import pandas as pd
data = pd.read_csv('merged_data.csv')
max_upc = 0
max_length = 0
for upc in set(data.UPC):
    data_selected = data[data.UPC == upc]
    print(upc)
    print('len of upc{}'.format(upc), len(data_selected))
    if len(data_selected) > max_length:
        max_length = len(data_selected)
        max_upc = upc
#data_selected = data_selected.last().sort_values('WEEK_END_DATE')
#print('data length', len(data_selected))
data_upc = data[data.UPC == max_upc]
data_upc.to_csv('filtered_dataset.csv')