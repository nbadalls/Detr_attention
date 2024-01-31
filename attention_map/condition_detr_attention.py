import torch, math
from torch.nn.functional import softmax
from get_attentions import plot_multi_head_feature_map, GetAttention


class ConditionalDetr(GetAttention):
    def define_model(self):
        self.model = torch.hub.load('Atten4Vis/ConditionalDETR:main', 'conditional_detr_resnet50', pretrained=True)

    def multi_head_attention(self, q, k, weight, bias, dim):
        q = q.view(-1, 1 * 8, dim).transpose(1, 0)  # 8x100x32
        k = k.view(-1, 1 * 8, dim).transpose(1, 0)  # 8xnx32
        attn_output_weights = torch.bmm(q, k.transpose(-1, -2))  # 8x100xn
        attn_output_weights /= math.sqrt(dim)
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        return attn_output_weights[None]

    def hook_qk_attention(self, key_type="combine"):
        """
        key_type: position, content, combine
        """
        assert key_type in ["position", "content", "combine"]

        q_content, k_content, qpos_sine, k_pos, pos = [], [], [], [], []

        # 作者自己实现了Multihead的attention操作，将qkv高维度的映射放到了函数之外进行（ca_qcontent_proj、ca_kcontent_proj、ca_kpos_proj）
        # 这个部分跟detr的实现有所区别
        hooks = [
            self.model.transformer.decoder.layers[-1].ca_qcontent_proj.register_forward_hook(
                lambda self, input, output: q_content.append(output)),
            self.model.transformer.decoder.layers[-1].ca_kcontent_proj.register_forward_hook(
                lambda self, input, output: k_content.append(output)),
            self.model.transformer.decoder.layers[-1].ca_kpos_proj.register_forward_hook(
                lambda self, input, output: k_pos.append(output)),
            self.model.transformer.decoder.layers[-1].ca_qpos_sine_proj.register_forward_hook(
                lambda self, input, output: qpos_sine.append(output)),
            self.model.backbone.register_forward_hook(
                lambda self, input, output: pos.append(output[1])),
        ]

        keep, bboxes_scaled, probas, im = self.model_inference()
        for hook in hooks:
            hook.remove()

        q_content, k_content, qpos_sine, k_pos, pos = q_content[0], k_content[0], qpos_sine[0], k_pos[0], pos[0]
        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape
        _, h, w = pos[-1].shape[-3:]  # 256 h w

        # 8个头分别进行content+position的维度拼接
        q_content = q_content.view(num_queries, bs, 8, n_model // 8) # from tgt
        query_sine_embed = qpos_sine.view(num_queries, bs, 8, n_model // 8) # from query_pos, MLP to position bias
        k_content = k_content.view(hw, bs, 8, n_model // 8) # from memory
        k_pos = k_pos.view(hw, bs, 8, n_model // 8)  # from backbone position embedding
        if key_type == "combine":
            k = torch.cat([k_content, k_pos], dim=3)
            q = torch.cat([q_content, query_sine_embed], dim=3)
        elif key_type == "content":
            q = q_content
            k = k_content
        elif key_type == "position":
            q = query_sine_embed
            k = k_pos

        dim = q.shape[-1]
        attn_output_weights = self.multi_head_attention(q, k, None, None, dim)
        plot_multi_head_feature_map(h, w, bboxes_scaled, attn_output_weights, keep, im, probas)