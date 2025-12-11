import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
import copy
import gc
import math


class LLMFeatureExtractor(torch.nn.Module):
    def __init__(self, conversation_template, model_name="deepseek-ai/deepseek-vl2-tiny", layer1_idx=0, layer2_idx=-1):
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

        self.conversation_template = conversation_template

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
    def __init__(self, layers, model_name="deepseek-ai/deepseek-vl2-tiny"):
        """
        Args:
            layers (list or tuple): Indices of layers to extract features from.
            model_name (str): Path to the pretrained DeepSeek-VL2 model (e.g., "deepseek-ai/deepseek-vl2-tiny").
                              NOTE: This must be a DeepSeek-VL2 model path, not a SigLIP config name.
        """
        super().__init__()
        
        self.layers = layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the full DeepSeek-VL2 model to access the pretrained vision tower
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Extract the Vision Tower and discard the rest
        self.fe = full_model.vision
        self.fe.to(self.device).eval()

        # Attributes required by your usage
        self.patch_size = self.fe.patch_embed.patch_size[0]
        self.embed_dim = self.fe.embed_dim

        # Cleanup LLM parts to save memory
        del full_model.language
        del full_model.projector
        del full_model
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, 384, 384).
                              Should be normalized using the processor's mean/std.
        """
        x = x.to(self.device, dtype=torch.bfloat16)
        
        return self.fe.get_intermediate_layers(x, n=self.layers)
    


# ====================================== #
#           FE FOR HD IMAGES             #
# ====================================== #


class HDLLMFeatureExtractor(torch.nn.Module):
    def __init__(self, layers=[0, 1], model_name="deepseek-ai/deepseek-vl2-tiny"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layers = sorted(layers)
        
        # Load Processor
        self.processor = DeepseekVLV2Processor.from_pretrained(model_name)
        
        # Load Full Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        # Config extraction for dynamic tiling calculations
        vision_cfg = self.model.config.vision_config
        projector_cfg = self.model.config.projector_config
        
        # Calculate the feature map size of a single tile
        self.tile_h = math.ceil((vision_cfg.image_size // vision_cfg.patch_size) / projector_cfg.downsample_ratio)
        self.tile_w = self.tile_h
        self.embed_dim = self.model.config.language_config.hidden_size

    @torch.no_grad()
    def forward(self, pil_images):
        """
        Args:
            pil_images: List of PIL Images.
        Returns:
            earlier_feat: [Batch, HD_H, HD_W, Dim]
            later_feat:   [Batch, HD_H, HD_W, Dim]
        """
        if not isinstance(pil_images, list):
            pil_images = [pil_images]

        # 1. Process Images & Prepare Batch
        # We process one by one to capture the grid info, then batchify
        processed_outputs = []
        grid_sizes = [] # Store (w_tiles, h_tiles) for each image
        
        for image in pil_images:
            # Generate tiles (Global + Locals)
            proc_out = self.processor.process_one(
                prompt="<image>", images=[image], inference_mode=True
            )
            processed_outputs.append(proc_out)
            grid_sizes.append(proc_out.images_spatial_crop[0].tolist())

        # Pads sequences and stacks images
        prepare_inputs = self.processor.batchify(processed_outputs).to(self.device, dtype=torch.bfloat16)

        # 2. Forward Pass through LLM
        # This converts images to embeddings and stitches them
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs['attention_mask'],
            output_hidden_states=True,
            use_cache=False
        )

        # 3. Extract & Clean Features
        batch_earlier = []
        batch_later = []
        
        # Get hidden states for requested layers
        # outputs.hidden_states is a tuple of tensors [Batch, Seq_Len, Dim]
        feat_early_raw = outputs.hidden_states[self.layers[0]]
        feat_late_raw = outputs.hidden_states[self.layers[1]]

        for i in range(len(pil_images)):
            # Identify Visual Tokens
            # images_seq_mask tells us where the visual tokens are in the sequence
            seq_mask = prepare_inputs['images_seq_mask'][i]
            
            # Extract only the visual part of the sequence (Global + Separator + Local)
            vis_early = feat_early_raw[i, seq_mask, :]
            vis_late = feat_late_raw[i, seq_mask, :]
            
            # B. Get Grid Info for this specific image
            w_tiles, h_tiles = grid_sizes[i]
            
            # C. Calculate lengths
            # Global View: (Tile_H) * (Tile_W + 1)  <-- +1 for newline
            global_len = self.tile_h * (self.tile_w + 1)
            # Separator: 1 token
            separator_len = 1
            
            # Start of Local Tokens = Global + Separator
            start_local = global_len + separator_len
            
            # D. Extract Local Tokens (The HD Map)
            local_early = vis_early[start_local:]
            local_late = vis_late[start_local:]
            
            # E. Reshape & Remove Newlines
            # DeepSeek stitches locals into: [Total_H, Total_W + 1] 
            # where Total_W = w_tiles * tile_w
            total_h = h_tiles * self.tile_h
            total_w = w_tiles * self.tile_w
            row_width = total_w + 1 # +1 for the newline token at end of row
            
            # Reshape: [Total_H, Total_W + 1, Dim]
            hd_early = local_early.view(total_h, row_width, -1)
            hd_late = local_late.view(total_h, row_width, -1)
            
            # Drop the last column (the newline token)
            # Result: [Total_H, Total_W, Dim]
            hd_early = hd_early[:, :-1, :]
            hd_late = hd_late[:, :-1, :]
            
            batch_earlier.append(hd_early)
            batch_later.append(hd_late)

        return torch.stack(batch_earlier), torch.stack(batch_later)
    

class HDViTFeatureExtractor(torch.nn.Module):
    def __init__(self, layers=[21, 25], model_name="deepseek-ai/deepseek-vl2-tiny"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layers = sorted(layers)
        
        # Load Processor
        self.processor = DeepseekVLV2Processor.from_pretrained(model_name)

        # Load Full Model
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Extract Vision Tower
        self.fe = full_model.vision
        self.fe.to(self.device).eval()
        
        # Get patch info (e.g., 14 or 16)
        self.patch_size = self.fe.patch_embed.patch_size[0] 
        self.embed_dim = self.fe.embed_dim
        
        # Cleanup
        del full_model.language
        del full_model.projector
        del full_model
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, pil_images):
        """
        Args:
            pil_images: List of PIL Images.
        Returns:
            earlier_feat: [Batch, HD_Grid_H, HD_Grid_W, Dim]
            later_feat:   [Batch, HD_Grid_H, HD_Grid_W, Dim]
        """
        if not isinstance(pil_images, list):
            pil_images = [pil_images]

        batch_earlier = []
        batch_later = []
        
        for image in pil_images:
            # DeepSeek Processing in Tiles
            proc_out = self.processor.process_one(
                prompt="<image>", images=[image], inference_mode=True
            )
            
            tiles = proc_out.images.to(self.device, dtype=torch.bfloat16)
            
            # Forward ViT (Output: Tuple of [Num_Tiles, Seq_Len, Dim])
            features_list = self.fe.get_intermediate_layers(tiles, n=self.layers)
            
            # Stitch Tiles into HD Map
            grid_w, grid_h = proc_out.images_spatial_crop[0].tolist()
            hd_early = self._stitch_to_hd_map(features_list[0], grid_h, grid_w)
            hd_late = self._stitch_to_hd_map(features_list[1], grid_h, grid_w)
            
            batch_earlier.append(hd_early)
            batch_later.append(hd_late)
            
        return torch.stack(batch_earlier), torch.stack(batch_later)

    def _stitch_to_hd_map(self, features, h_tiles, w_tiles):
        """
        Converts [Num_Tiles, Seq_Len, Dim] -> [HD_H, HD_W, Dim]
        Discards the Global View (Index 0).
        """
        # Drop Global View
        local_feats = features[1:]
        
        # Determine Patch Grid Dimensions
        n_locals, seq_len, dim = local_feats.shape
        side_patches = int(math.sqrt(seq_len))
        
        local_feats = local_feats.view(n_locals, side_patches, side_patches, dim)
        
        grid = local_feats.view(h_tiles, w_tiles, side_patches, side_patches, dim)
        
        grid = grid.permute(0, 2, 1, 3, 4).contiguous() # [H_tiles, Side, W_tiles, Side, Dim]
        
        # Collapse to HD Map: [H_total, W_total, Dim]
        hd_h = h_tiles * side_patches
        hd_w = w_tiles * side_patches
        hd_map = grid.view(hd_h, hd_w, dim)
        
        return hd_map
    

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import torch

    dummy_image = Image.fromarray(np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8))

    layers_to_extract = [0, 1]
    extractor = HDLLMFeatureExtractor(layers=layers_to_extract)

    results = extractor([dummy_image])
    
    print("\n=== Output Dimension Check ===")
    for i, img_features in enumerate(results):
        print(f" Layers {i} shape: {tuple(img_features.shape)}")