import torch
from collections import OrderedDict

def convert_state_dict(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        if k.endswith('.attn.qkv.bias'):
            prefix = k[:-len('.qkv.bias')]
            qkv_bias = v
            q_bias, k_bias, v_bias = qkv_bias.chunk(3)
            new_state_dict[f'{prefix}.q_bias'] = q_bias
            new_state_dict[f'{prefix}.v_bias'] = v_bias
        else:
            new_state_dict[k] = v
    return new_state_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert new checkpoint to old format')
    parser.add_argument('input', type=str, help='Path to the input checkpoint file')
    args = parser.parse_args()

    epoch = args.input.split('-')[-1].replace('.pth', '')
    output = args.input.split("/")[-2] + f'_e{epoch}.pth'

    print(f'Converting {args.input} to {output}')

    # Load checkpoint
    checkpoint = torch.load(args.input, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = convert_state_dict(state_dict)

    # Save the converted checkpoint, preserving other keys
    if isinstance(checkpoint, dict):
        checkpoint['model'] = new_state_dict
        torch.save(checkpoint, output)
    else:
        torch.save({'model': new_state_dict}, output)


if __name__ == '__main__':
    main()