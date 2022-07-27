from datasets_for_transfer import *

data_path = '/dataset'
mode = 'val'

dataloader = ImageDataset(data_path, mode=mode).set_attrs(
    batch_size=1,
    shuffle=False,
    num_workers=1,
    one_hot_label=False
)

val_semantics_list = []

unique_values_list = []

for i, (_, sem, photo_id) in enumerate(dataloader):
    cur_semantics = []
    unique_values = jt.unique(sem)
    for v in unique_values:
        v = int(v.data)
        cur_semantics.append(v)
        if v not in unique_values_list:
            unique_values_list.append(v)
    val_semantics_list.append((photo_id,cur_semantics))

    print("processing:", i)


import pickle
with open('val_semantics_list.data', 'wb') as f:
    pickle.dump(val_semantics_list, f)

with open('val_semantics_list.data', 'rb') as f:
    val_semantics_list = pickle.load(f)

