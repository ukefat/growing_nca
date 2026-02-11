import torch

def compute_mean_var_by_part(cell_states):
    
    def ragged_stat(cell_states, stat="mean"):
        stats = []
        for state in cell_states:
            if state.shape[1] > 0:
                if stat == "mean":
                    s = state.mean(dim=1)
                elif stat == "var":
                    s = state.var(dim=1, unbiased=False)
                else:
                    raise ValueError("stat must be 'mean' or 'var'")
            else:
                s = torch.zeros(state.shape[0])
            stats.append(s)
        return torch.stack(stats)

    return {
        "means": {
            "image": cell_states['image'].mean(dim=2),
            "alive": ragged_stat(cell_states['alive'], "mean"),
            "eyes": ragged_stat(cell_states['eyes'], "mean"),
            "tongue": ragged_stat(cell_states['tongue'], "mean"),
            "body": ragged_stat(cell_states['body'], "mean"),
            "r_eye": ragged_stat(cell_states['r_eye'], "mean"),
            "l_eye": ragged_stat(cell_states['l_eye'], "mean"),
        },
        "vars": {
            "image": cell_states['image'].var(dim=2, unbiased=False),
            "alive": ragged_stat(cell_states['alive'], "var"),
            "eyes": ragged_stat(cell_states['eyes'], "var"),
            "tongue": ragged_stat(cell_states['tongue'], "var"),
            "body": ragged_stat(cell_states['body'], "var"),
            "r_eye": ragged_stat(cell_states['r_eye'], "var"),
            "l_eye": ragged_stat(cell_states['l_eye'], "var"),
        }
    }

def get_eye_states(states, side="r"):
    t,c,h,w = states.shape

    if side == "r":
        states_half = states[:, :, :, w//2:]
    else:
        states_half = states[:, :, :, 0:w//2]
    
    rgb = states_half[:, 0:3] # (t, 3, p)
    alpha = states_half[:, 3] # (t, p)
    
    eye_mask = ((rgb < 0.1).all(dim=1) & 
                    (alpha > 0.9))
    eye_states = [
        states_half[t, :, eye_mask[t]]
        for t in range(states_half.shape[0])
    ]
    return eye_states

def get_states_by_part(states):

    t,c,h,w = states.shape

    a_thresh = 0.9
    
    states_flat = states.view(states.shape[0], states.shape[1], -1) # (t, c, pixels)
    rgb = states_flat[:, 0:3] # (t, 3, p)
    alpha = states_flat[:, 3] # (t, p)
    
    alive_mask = states_flat[:, 3] > a_thresh   # (t, pixels)
    eye_mask = ((rgb < 0.1).all(dim=1) & 
                (alpha > a_thresh))
    body_mask = ((rgb[:, 0] < 0.1) & (rgb[:, 1] > 0.9) & (rgb[:, 2] < 0.1) &
                 (alpha > a_thresh))
    tongue_mask = ((rgb[:, 0] > 0.9) & (rgb[:, 1] < 0.1) & (rgb[:, 2] > 0.9) &
                   (alpha > a_thresh))

    def get_states(mask):
        masked_states = [
            states_flat[t, :, mask[t]]
            for t in range(states.shape[0])
        ]# (t, c, alive_pixels)
        return masked_states

    alive_states = get_states(alive_mask)
    eye_states = get_states(eye_mask)
    body_states = get_states(body_mask)
    tongue_states = get_states(tongue_mask)

    return {
        "image": states_flat,
        "alive": alive_states,
        "eyes": eye_states,
        "body": body_states,
        "tongue": tongue_states,
        "r_eye": get_eye_states(states, "r"),
        "l_eye": get_eye_states(states, "l"),
    }
