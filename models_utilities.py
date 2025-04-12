

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def rename_module_prefix(state_dict, old_name, new_name):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix
        new_key = key.replace(old_name, new_name)
        # encoder.embeddings.patch_embeddings.projection.weight
        # model.videomae.embeddings.patch_embeddings.projection.weight
        new_state_dict[new_key] = value
    return new_state_dict

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total_params: {total_params}, Trainable_params: {trainable_params}")
    return total_params, trainable_params