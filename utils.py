

def get_dummy_variable(image_name):
    types = {
        'Star': 0,
        'Square': 1,
        'Circle': 2,
        'Cross': 3,
        'Triangle': 4,
        '1': 5,
        '2': 6,
        '3': 7,
        '4': 8,
        '5': 9,
        'Blue': 10,
        'Green': 11,
        'Red': 12,
        'Yellow': 13,
        'Black': 14,
    }
    dv = [0] * len(types)
    for k, v in types.items():
        if k in image_name:
            dv[v] = 1
    return dv

def get_dummy_variables(image_list):
    dvs = []
    for image_name in image_list:
        tmp = get_dummy_variable(image_name)
        dvs.append(tmp)
    return dvs

def get_q_by_t(state, t_idx):
    rewards = []
    for s in state:
        if s[t_idx] > 0.5:
            rewards.append([1., -1.])
        else:
            rewards.append([-1., 1.])
    return rewards

def get_t_by_idx(state, t_idx):
    rewards = []
    for s in state:
        if s[t_idx] > 0.5:
            rewards.append([1.])
        else:
            rewards.append([0.])
    return rewards
