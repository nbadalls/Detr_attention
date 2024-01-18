from attention_map import get_attentions



if __name__ == "__main__":
    atten = get_attentions.GetAttention()
    atten.hook_qk_attention("combine")