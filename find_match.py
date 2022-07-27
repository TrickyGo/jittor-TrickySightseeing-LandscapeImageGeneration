import pickle
with open('training_semantics_list.data', 'rb') as f:
    training_semantics_list = pickle.load(f)
with open('val_semantics_list.data', 'rb') as f:
    val_semantics_list = pickle.load(f)

match_dict = {} #{val_index: [training_index, ...]}
no_match_cnt = 0
for val_index in range(1000):
    print("searching for:",val_index)
    cur_val_semantics = val_semantics_list[val_index][1]
    matching_list = []
    for training_index in range(10000):
        cur_training_semantics = training_semantics_list[training_index][1]

        is_match = True
        for v in cur_val_semantics:
            if v not in cur_training_semantics:
                is_match = False 

        if is_match:
            matching_list.append(training_index)

    if len(matching_list) == 0:
        cur_val_semantics = cur_val_semantics[:-5]

        for training_index in range(10000):
            cur_training_semantics = training_semantics_list[training_index][1]

            is_match = True
            for v in cur_val_semantics:
                if v not in cur_training_semantics:
                    is_match = False 

            if is_match:
                matching_list.append(training_index)

    if len(matching_list) == 0:
        no_match_cnt += 1 

    match_dict[val_index] = matching_list


with open('match_dict.data', 'wb') as f:
    pickle.dump(match_dict, f)    
