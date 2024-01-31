from attention_map import get_attentions
from attention_map import condition_detr_attention
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('plot attention', add_help=False)
    parser.add_argument('--type', default="detr", type=str, help="detr cd_detr")
    parser.add_argument('--plot', default="combine", type=str, help="combine content position")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.type == "detr":
        atten = get_attentions.GetAttention()
    elif  args.type == "cd_detr":
        atten = condition_detr_attention.ConditionalDetr()
    else:
        print("wrong type!")
        exit(0)
    atten.hook_qk_attention(args.plot)