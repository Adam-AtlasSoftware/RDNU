import torch
import argparse
from collections import OrderedDict

def get_new_shape(name, shape, old_dim, new_dim):
    new_shape = list(shape)
    
    multipliers = [1, 2, 3, 4, 6, 8, 12, 16]
    fractions = [0.25, 0.5, 0.75, 1.5, 2.5]
    
    for i, s in enumerate(shape):
        matched = False
        
        # Protect specific layers that shouldn't have out_channels scaled
        if 'UpSample' in name and i == 0:
            continue
            
        for m in multipliers:
            if s == old_dim * m:
                new_shape[i] = new_dim * m
                matched = True
                break
        
        if not matched:
            for m in multipliers:
                if s == (old_dim * m) + 1:
                    new_shape[i] = (new_dim * m) + 1
                    matched = True
                    break
        
        if not matched and 'alpha' in name:
            for f in fractions:
                if s == int(old_dim * f):
                    new_shape[i] = int(new_dim * f)
                    matched = True
                    break
                    
    return tuple(new_shape)

def inflate_weights(input_path, output_path, old_dim=16, new_dim=36):
    print(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
        ema_dict = checkpoint.get('params_ema', None)
    else:
        state_dict = checkpoint
        ema_dict = None
        
    def pad_tensor(name, tensor, old_dim, new_dim):
        shape = list(tensor.shape)
        new_shape = get_new_shape(name, shape, old_dim, new_dim)
        
        if tuple(shape) == tuple(new_shape):
            return tensor
            
        new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
        
        slices = []
        for s_new, s_old in zip(new_shape, shape):
            slices.append(slice(0, s_old))
            
        if 'float' in str(tensor.dtype):
            torch.nn.init.normal_(new_tensor, mean=0.0, std=0.01)
            
        new_tensor[tuple(slices)] = tensor
        return new_tensor

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k] = pad_tensor(k, v, old_dim, new_dim)
        if new_state_dict[k].shape != v.shape:
            print(f"Inflated {k}: {v.shape} -> {new_state_dict[k].shape}")
            
    if ema_dict is not None:
        new_ema_dict = OrderedDict()
        for k, v in ema_dict.items():
            new_ema_dict[k] = pad_tensor(k, v, old_dim, new_dim)
    else:
        new_ema_dict = None

    out_dict = {}
    if 'params' in checkpoint:
        out_dict['params'] = new_state_dict
        if new_ema_dict is not None:
            out_dict['params_ema'] = new_ema_dict
    else:
        out_dict = new_state_dict
        
    torch.save(out_dict, output_path)
    print(f"Successfully saved inflated weights to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--old_dim', type=int, default=16)
    parser.add_argument('--new_dim', type=int, default=36)
    args = parser.parse_args()
    
    inflate_weights(args.input, args.output, args.old_dim, args.new_dim)
