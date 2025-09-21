import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, siglip_vit
import PIL
from typing import List, Tuple
import copy

class LLMFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name="deepseek-ai/deepseek-vl2-tiny", layer1_idx=0, layer2_idx=-1):
        super().__init__()

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(torch.bfloat16).to(self.device).eval()

        num_layers = self.model.config.language_config.num_hidden_layers
        total_states = num_layers + 1
        assert -total_states <= layer1_idx < total_states
        assert -total_states <= layer2_idx < total_states
        self.layer1_idx = layer1_idx
        self.layer2_idx = layer2_idx
        self.embed_dim = self.model.config.language_config.hidden_size
        self.num_global_patches = 14 * 14

        prompt_text = "This image shows <|ref|>industrial products in perfect condition that meet all quality standards<|/ref|>."
        self.conversation_template = [
            {"role": "<|User|>", "content": f"<image>\n{prompt_text}"},
            {"role": "<|Assistant|>", "content": ""}
        ]

    @torch.no_grad()
    def forward(self, pil_images):
        processed_outputs = []
        for image in pil_images:
            conversation = copy.deepcopy(self.conversation_template)
            processed_output = self.processor.process_one(
                conversations=conversation,
                images=[image]
            )
            processed_outputs.append(processed_output)

        # Assembla il batch
        prepare_inputs = self.processor.batchify(processed_outputs).to(self.device)

        # Esegui il modello sull'intero batch
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs['attention_mask'],
            output_hidden_states=True,
            use_cache=False
        )

        # Estrazione e Filtraggio
        hidden_states = outputs.hidden_states
        features1 = hidden_states[self.layer1_idx]
        features2 = hidden_states[self.layer2_idx]

        image_seq_mask = prepare_inputs['images_seq_mask'][0]

        all_visual_embeds1 = features1[:, image_seq_mask, :]
        all_visual_embeds2 = features2[:, image_seq_mask, :]
        input_embeds_visual = hidden_states[0][:, image_seq_mask, :]

        newline_embedding = self.model.image_newline
        separator_embedding = self.model.view_seperator

        is_newline_token = torch.all(input_embeds_visual == newline_embedding, dim=2)
        is_separator_token = torch.all(input_embeds_visual == separator_embedding, dim=2)

        pure_patch_mask_1d = ~(is_newline_token[0] | is_separator_token[0])

        pure_embeddings1 = all_visual_embeds1[:, pure_patch_mask_1d, :]
        pure_embeddings2 = all_visual_embeds2[:, pure_patch_mask_1d, :]

        final_global_view1 = pure_embeddings1[:, :self.num_global_patches, :]
        final_global_view2 = pure_embeddings2[:, :self.num_global_patches, :]

        return final_global_view1, final_global_view2


class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, layers, model_name="siglip_so400m_patch14_384", image_size=384):
        super().__init__()

        self.fe = siglip_vit.create_siglip_vit(model_name=model_name, image_size=image_size)
        self.layers = layers

        self.patch_size = self.fe.patch_embed.patch_size[0]
        self.embed_dim = self.fe.embed_dim

    def forward(self, x):
        return self.fe.get_intermediate_layers(x, n=self.layers)