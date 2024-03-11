import sys
sys.path.append("./attention_map")
import torch
from utility import transform, rescale_bboxes, plot_multi_head_feature_map
from PIL import Image
import requests
from torch.nn.functional import linear,softmax
torch.set_grad_enabled(False)
import torch, math

class GetAttention:
    def __init__(self):
        self.model = None
        self.define_model()

    def input_image(self):
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        im = Image.open(requests.get(url, stream=True).raw)
        # mean-std normalize the input image (batch-size: 1)
        trans_img = transform(im).unsqueeze(0)
        return im, trans_img

    def define_model(self):
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

    def model_inference(self):
        self.model.eval()
        im, trans_img = self.input_image()
        ret = self.model(trans_img)

        probas = ret["pred_logits"].softmax(-1)[0, :, :-1]  # 100x92
        keep = probas.max(-1).values > 0.5
        bboxes_scaled = rescale_bboxes(ret["pred_boxes"][0, keep], im.size)  # 1x100x4
        return keep, bboxes_scaled, probas, im

    def multi_head_attention(self, q, k, weight, bias, dim):
        head_dim = dim // 8
        q = linear(q, weight[:dim, :], bias[:dim])  # 100x1x256
        k = linear(k, weight[dim:2 * dim, :], bias[dim:2 * dim])  # nx1x256
        q = q.view(-1, 1 * 8, head_dim).transpose(1, 0)  # 8x100x32
        k = k.view(-1, 1 * 8, head_dim).transpose(1, 0)  # 8xnx32
        attn_output_weights = torch.bmm(q, k.transpose(-1, -2))  # 8x100xn
        attn_output_weights /= math.sqrt(head_dim)
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        return attn_output_weights[None]

    def hook_qk_attention(self, key_type="combine"):
        """
        key_type: position, content, combine
        """
        assert key_type in ["position", "content", "combine"]

        tgt, memory = [], []
        query_pos, pos = self.model.query_embed.weight, []

        qkv_weight = self.model.transformer.decoder.layers[-1].multihead_attn.in_proj_weight
        qkv_bias = self.model.transformer.decoder.layers[-1].multihead_attn.in_proj_bias
        hooks = [
            self.model.transformer.decoder.layers[-1].norm1.register_forward_hook(
                lambda self, input, output: tgt.append(output)),
            self.model.transformer.encoder.register_forward_hook(
                lambda self, input, output: memory.append(output)),
            self.model.backbone.register_forward_hook(
                lambda self, input, output: pos.append(output[1])),
        ]
        keep, bboxes_scaled, probas, im = self.model_inference()
        for hook in hooks:
            hook.remove()

        pos, tgt, memory = pos[0], tgt[0], memory[0]
        dim, h, w = pos[-1].shape[-3:]  # 256 h w
        pos_embed = pos[-1].flatten(2).permute(2, 0, 1)  # n x 1 x hxw
        query = tgt + query_pos[:, None, :]  # 1x100x256
        key = memory + pos_embed  # 1xnx256
        if key_type == "position":
            query = query_pos[:, None, :]
            key = pos_embed
        elif key_type == "content":
            query = tgt
            key = memory

        # multi_attention = nn.MultiheadAttention(dim, 8, dropout=0)
        # multi_attention.in_proj_bias = qkv_bias
        # multi_attention.in_proj_weight = qkv_weight
        # _, attn_output_weights = multi_attention(query, key, memory,
        #                                         average_attn_weights=False)
        # 手动实现attention计算，近似等于torch函数计算结果
        attn_output_weights = self.multi_head_attention(query, key, qkv_weight, qkv_bias, dim)
        plot_multi_head_feature_map(h, w, bboxes_scaled, attn_output_weights, keep, im, probas)