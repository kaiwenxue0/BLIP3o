from abc import ABC, abstractmethod

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower, build_gen_vision_tower, build_dit
from .multimodal_projector.builder import build_vision_projector, build_down_projector, build_gen_vision_projector

from blip3o.constants import IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_IDX, UND_IMAGE_TOKEN_IDX, DEFAULT_IMAGE_PATCH_TOKEN, IMAGE_TOKEN_IDX_LLADA
from llava.model.multimodal_resampler.builder import build_vision_resampler
# TODO DEFAULT_IM_START_TOKEN_LLaDA not used
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN as DEFAULT_IM_START_TOKEN_LLaDA, DEFAULT_IM_END_TOKEN as DEFAULT_IM_END_TOKEN_LLaDA
from llava.utils import rank0_print


class blip3oMetaModel:

    def __init__(self, config):
        super(blip3oMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # self.vision_tower = build_vision_tower(config, delay_load=True)
            # self.mm_projector = build_vision_projector(config)
            self.down_projector = build_down_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )


        if hasattr(config, "gen_vision_tower"):
            self.gen_vision_tower = build_gen_vision_tower(config, delay_load=True)
            # self.gen_projector = build_gen_vision_projector(config)
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
            print(f" latent query size {self.latent_queries.shape}")

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

            self.dit, self.noise_scheduler = build_dit(config)


    # def get_vision_tower(self):
    #     vision_tower = getattr(self, 'vision_tower', None)
    #     if type(vision_tower) is list:
    #         vision_tower = vision_tower[0]
    #     return vision_tower


    def get_gen_vision_tower(self):
        gen_vision_tower = getattr(self, 'gen_vision_tower', None)
        if type(gen_vision_tower) is list:
            gen_vision_tower = gen_vision_tower[0]
        return gen_vision_tower


    def initialize_vision_modules(self, model_args, fsdp=None):
        gen_vision_tower = model_args.gen_vision_tower

        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_gen_mlp_adapter = model_args.pretrain_gen_mlp_adapter

        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.gen_vision_tower = gen_vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")



        if getattr(self, 'dit', None) is None:
            print("random initiation the DiT !!!")
            self.dit, self.noise_scheduler = build_dit(model_args)
        else:
            print("DiT load from checkpoint!!!")
            for p in self.dit.parameters():
                p.requires_grad = True
    

        if self.get_gen_vision_tower() is None:
            gen_vision_tower = build_gen_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.gen_vision_tower = [gen_vision_tower]
            else:
                self.gen_vision_tower = gen_vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                gen_vision_tower = self.gen_vision_tower[0]
            else:
                gen_vision_tower = self.gen_vision_tower
            gen_vision_tower.load_model()


        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        # self.config.gen_projector_type = getattr(model_args, 'gen_projector_type', 'linear')


        self.config.gen_hidden_size = gen_vision_tower.hidden_size

        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.n_query = model_args.n_query
        self.config.gen_pooling = model_args.gen_pooling

        # if getattr(self, 'mm_projector', None) is None:
        #     print("random initiation the mm_project !!!")
        #     self.mm_projector = build_vision_projector(self.config)

        #     if 'unpad' in mm_patch_merge_type:
        #         embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
        #         self.image_newline = nn.Parameter(
        #             torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
        #         )
        # else:
        #     # In case it is frozen by LoRA
        #     for p in self.mm_projector.parameters():
        #         p.requires_grad = True



        if getattr(self, 'down_projector', None) is None:
            self.down_projector = build_down_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.down_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'latent_queries', None) is None:
            print("random initiation the latent_queries !!!")
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))
        else:
            print("latent_queries load from checkpoint!!!")
            self.latent_queries.requires_grad = True


        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class blip3oMetaModelForLLaDA:

    def __init__(self, config):
        super(blip3oMetaModelForLLaDA, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            self.down_projector = build_down_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )


        if hasattr(config, "gen_vision_tower"):
            self.gen_vision_tower = build_gen_vision_tower(config, delay_load=True)
            # self.gen_projector = build_gen_vision_projector(config)
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
            print(f" latent query size {self.latent_queries.shape}")

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

            self.dit, self.noise_scheduler = build_dit(config)


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


    def get_gen_vision_tower(self):
        gen_vision_tower = getattr(self, 'gen_vision_tower', None)
        if type(gen_vision_tower) is list:
            gen_vision_tower = gen_vision_tower[0]
        return gen_vision_tower


    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        gen_vision_tower = model_args.gen_vision_tower

        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_gen_mlp_adapter = model_args.pretrain_gen_mlp_adapter

        mm_patch_merge_type = model_args.mm_patch_merge_type
        
        self.config.mm_vision_tower = vision_tower
        self.config.gen_vision_tower = gen_vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        if getattr(self, 'dit', None) is None:
            print("random initiation the DiT !!!")
            self.dit, self.noise_scheduler = build_dit(model_args)
        else:
            print("DiT load from checkpoint!!!")
            for p in self.dit.parameters():
                p.requires_grad = True
    

        if self.get_gen_vision_tower() is None:
            gen_vision_tower = build_gen_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.gen_vision_tower = [gen_vision_tower]
            else:
                self.gen_vision_tower = gen_vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                gen_vision_tower = self.gen_vision_tower[0]
            else:
                gen_vision_tower = self.gen_vision_tower
            gen_vision_tower.load_model()


        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        
        self.config.gen_hidden_size = gen_vision_tower.hidden_size

        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.n_query = model_args.n_query
        self.config.gen_pooling = model_args.gen_pooling

        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


        if getattr(self, 'down_projector', None) is None:
            self.down_projector = build_down_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.down_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'latent_queries', None) is None:
            print("random initiation the latent_queries !!!")
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))
        else:
            print("latent_queries load from checkpoint!!!")
            self.latent_queries.requires_grad = True


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class blip3oMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_gen_vision_tower(self):
        return self.get_model().get_gen_vision_tower()

    def encode_image(self, images):
        # breakpoint()
        gen_vision_tower = self.get_gen_vision_tower()
        device = gen_vision_tower.device
        images = images.to(device)
        prompt_image_embeds = gen_vision_tower(images)
        if 'early' in self.get_gen_pooling():
            prompt_image_embeds = self.pool_img(prompt_image_embeds)
        num_img, _, c = prompt_image_embeds.shape
        # prompt_image_embeds = prompt_image_embeds.contiguous().view(-1, c)

        # ------------- compute similarity -------
        all_dist = 0
        count = 0
        for i in range(2, prompt_image_embeds.shape[1]-1):
            diff = (prompt_image_embeds[:,i,:].unsqueeze(1) -  prompt_image_embeds[:,:i,:])
            dist = torch.sqrt(diff.square().sum(-1)).min().item()
            all_dist+=dist
            count+=1
        all_dist /= count
        # self.dist = all_dist
        # print(self.dist)

        return prompt_image_embeds

    def get_mm_projector(self):
        return self.get_model().mm_projector

    def get_gen_projector(self):
        return None
    

    def get_n_query(self):
        return self.get_model().config.n_query

    def get_gen_pooling(self):
        return self.get_model().config.gen_pooling

    def pool_img(self, image_features):
        num_img, n, c = image_features.shape
        gen_pooling = self.get_gen_pooling()
        # n_query = self.get_n_query()
        stride = int(gen_pooling.split('_')[-1])
        sqrt_n = int(n**0.5)
        image_features = image_features.permute(0, 2, 1).view(num_img, c, sqrt_n, sqrt_n)
        image_features = F.avg_pool2d(image_features, kernel_size=(stride, stride), stride=stride)
        # image_features = image_features.view(num_img, c, -1).permute(0,2,1).contiguous()
        return image_features

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gen_images, und_images, grid_thw, i_s_pos, image_sizes=None
    ):
        pad_ids = 128256
        vision_tower = self.visual
        gen_vision_tower = self.get_gen_vision_tower()
        if (gen_images is None and und_images is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None
        



        prompt_image_embeds = gen_vision_tower(gen_images) # TODO: check dimension
      
        if 'early' in self.get_gen_pooling():
            prompt_image_embeds = self.pool_img(prompt_image_embeds)
        target_image_embeds = torch.clone(prompt_image_embeds).detach()
        latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        H = latent_queries.shape[-1]
        latent_queries = latent_queries.contiguous().view(-1, H)
    

        # if not gen_images is None:
        #     prompt_image_embeds = gen_vision_tower(gen_images) # TODO: check dimension
        #     if 'early' in self.get_gen_pooling():
        #         prompt_image_embeds = self.pool_img(prompt_image_embeds)
        #     # num_img, _, c = prompt_image_embeds.shape  # [batch, 729, 1152]
        #     # prompt_image_embeds = prompt_image_embeds.contiguous().view(-1, c)
        #     target_image_embeds = torch.clone(prompt_image_embeds).detach()
        #     # prompt_image_embeds = gen_projector(prompt_image_embeds)
        #     latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        #     H = latent_queries.shape[-1]
        #     latent_queries = latent_queries.contiguous().view(-1, H)
        # else:
        #     target_image_embeds = None
        #     num_img = und_images.shape[0]
        #     dummy = torch.zeros(num_img, 3, 448, 448 , dtype=und_images.dtype, device=und_images.device) # TODO
        #     temp = gen_vision_tower(dummy)[:,:729,:]
        #     num_img, _, c = temp.shape
        #     temp = temp.contiguous().view(-1, c) * 1e-20
        #     # temp = gen_projector(temp) * 1e-9
        #     latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        #     H = latent_queries.shape[-1]
        #     latent_queries = latent_queries.contiguous().view(-1, H)


        if not und_images is None:
            und_image_embeds = vision_tower(und_images, grid_thw=grid_thw)
            _, c = und_image_embeds.shape
            batch_size = und_images.shape[0]
            
            # only for single input image per sample case
            und_image_embeds = und_image_embeds.view(batch_size, -1, c)

            # und_image_embeds = und_image_embeds.contiguous().view(-1, c)
            # und_image_embeds = mm_projector(und_image_embeds)

        # else:
        #     num_img = input_ids.shape[0]
        #     dummy = torch.zeros(num_img, 3, 384, 384 , dtype=gen_images.dtype, device=gen_images.device)  # clip (3, 336, 336) 
        #     temp = vision_tower(dummy)
        #     if 'early' in self.get_gen_pooling():
        #         temp = temp[:,:64,:]
        #     num_img, _, c = temp.shape
        #     temp = temp.contiguous().view(-1, c)
        #     temp = mm_projector(temp) * 1e-20
        #     latent_queries += temp



        
        image_idx = (input_ids == IMAGE_TOKEN_IDX)
        und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
        # img_indicator = torch.clone(image_idx)
        output_indicator = labels != -100
        input_indicator = labels == -100
        # img_loss_indicator = torch.logical_and(output_indicator, image_idx)
        # img_loss_indicator = torch.cat(
        #     [img_loss_indicator[:, 1:], img_loss_indicator[:, :1]], dim=1)
        
        # img_indicator = torch.cat(
        #     [img_indicator[:, 1:], img_indicator[:, :1]], dim=1)
        
        # if not target_image_embeds is None:
        #     target_image_embeds = target_image_embeds[-img_loss_indicator.sum():,:]
        text_embeds = self.get_model().embed_tokens(input_ids)
        # N_QUERY = self.get_n_query()
        gen_img_idx = torch.logical_and(output_indicator, image_idx)
       
        # if not target_image_embeds is None:
        text_embeds = text_embeds.clone() 
        text_embeds[gen_img_idx] = latent_queries
        # text_embeds[gen_img_idx] = prompt_image_embeds.to(text_embeds.device)[:gen_img_idx.sum(),:]
        # target_image_embeds = target_image_embeds.to(text_embeds.device)[:gen_img_idx.sum(),:]

        und_img_idx = torch.logical_and(input_indicator, und_image_idx)
     

        if not und_images is None:
            B, L, D = text_embeds.shape
            mask = und_img_idx.bool()

            for b in range(B):
                k = int(mask[b].sum().item())
                # 保障右侧够用
                assert und_image_embeds.size(1) >= k, f"sample {b}: need {k} embeds, got {und_image_embeds.size(1)}"
                text_embeds[b, mask[b]] = und_image_embeds[b, :k, :].to(text_embeds.device)

        labels[image_idx] = -100


        return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds



    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

class blip3oMetaForLlavaLLaDAModelLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_gen_vision_tower(self):
        return self.get_model().get_gen_vision_tower()

    def encode_image(self, images):
        # breakpoint()
        gen_vision_tower = self.get_gen_vision_tower()
        device = gen_vision_tower.device
        images = images.to(device)
        prompt_image_embeds = gen_vision_tower(images)
        if 'early' in self.get_gen_pooling():
            prompt_image_embeds = self.pool_img(prompt_image_embeds)
        num_img, _, c = prompt_image_embeds.shape
        # prompt_image_embeds = prompt_image_embeds.contiguous().view(-1, c)

        # ------------- compute similarity -------
        all_dist = 0
        count = 0
        for i in range(2, prompt_image_embeds.shape[1]-1):
            diff = (prompt_image_embeds[:,i,:].unsqueeze(1) -  prompt_image_embeds[:,:i,:])
            dist = torch.sqrt(diff.square().sum(-1)).min().item()
            all_dist+=dist
            count+=1
        all_dist /= count
        # self.dist = all_dist
        # print(self.dist)

        return prompt_image_embeds

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def get_mm_projector(self):
        return self.get_model().mm_projector

    def get_gen_projector(self):
        return None
    

    def get_n_query(self):
        return self.get_model().config.n_query

    def get_gen_pooling(self):
        return self.get_model().config.gen_pooling

    def pool_img(self, image_features):
        num_img, n, c = image_features.shape
        gen_pooling = self.get_gen_pooling()
        # n_query = self.get_n_query()
        stride = int(gen_pooling.split('_')[-1])
        sqrt_n = int(n**0.5)
        image_features = image_features.permute(0, 2, 1).view(num_img, c, sqrt_n, sqrt_n)
        image_features = F.avg_pool2d(image_features, kernel_size=(stride, stride), stride=stride)
        # image_features = image_features.view(num_img, c, -1).permute(0,2,1).contiguous()
        return image_features

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask

    def prepare_inputs_labels_for_multimodal_llada(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gen_images, und_images, grid_thw, i_s_pos, modalities=["image"], image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        gen_vision_tower = self.get_gen_vision_tower()
        if (gen_images is None and und_images is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None
        
        prompt_image_embeds = gen_vision_tower(gen_images) # TODO: check dimension
      
        if 'early' in self.get_gen_pooling():
            prompt_image_embeds = self.pool_img(prompt_image_embeds)
        target_image_embeds = torch.clone(prompt_image_embeds).detach()
        latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        H = latent_queries.shape[-1]
        latent_queries = latent_queries.contiguous().view(-1, H)


        # if not gen_images is None:
        #     prompt_image_embeds = gen_vision_tower(gen_images) # TODO: check dimension
        #     if 'early' in self.get_gen_pooling():
        #         prompt_image_embeds = self.pool_img(prompt_image_embeds)
        #     # num_img, _, c = prompt_image_embeds.shape  # [batch, 729, 1152]
        #     # prompt_image_embeds = prompt_image_embeds.contiguous().view(-1, c)
        #     target_image_embeds = torch.clone(prompt_image_embeds).detach()
        #     # prompt_image_embeds = gen_projector(prompt_image_embeds)
        #     latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        #     H = latent_queries.shape[-1]
        #     latent_queries = latent_queries.contiguous().view(-1, H)
        # else:
        #     target_image_embeds = None
        #     num_img = und_images.shape[0]
        #     dummy = torch.zeros(num_img, 3, 448, 448 , dtype=und_images.dtype, device=und_images.device) # TODO
        #     temp = gen_vision_tower(dummy)[:,:729,:]
        #     num_img, _, c = temp.shape
        #     temp = temp.contiguous().view(-1, c) * 1e-20
        #     # temp = gen_projector(temp) * 1e-9
        #     latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        #     H = latent_queries.shape[-1]
        #     latent_queries = latent_queries.contiguous().view(-1, H)


        if not und_images is None:
            if  type(und_images) is list or und_images.ndim == 5:
                if type(und_images) is list:
                    und_images = [x.unsqueeze(0) if x.ndim == 3 else x for x in und_images]

                video_idx_in_batch = []
                if modalities is not None:
                    for _ in range(len(modalities)):
                        if modalities[_] == "video":
                            video_idx_in_batch.append(_)

                images_list = []

                for image in und_images:
                    if image.ndim == 4:
                        images_list.append(image)
                    else:
                        images_list.append(image.unsqueeze(0))

                concat_images = torch.cat([image for image in images_list], dim=0)
                split_sizes = [image.shape[0] for image in images_list]
                encoded_image_features = self.encode_images(concat_images)
                # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

                # This is a list, each element is [num_images, patch * patch, dim]
                # rank_print(f"Concat images : {concat_images.shape}")
                encoded_image_features = torch.split(encoded_image_features, split_sizes)
                image_features = []
                for idx, image_feat in enumerate(encoded_image_features):
                    if idx in video_idx_in_batch:
                        image_features.append(self.get_2dPool(image_feat))
                    else:
                        image_features.append(image_feat)
                mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
                image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
                mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

                if mm_patch_merge_type == "flat":
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith("spatial"):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        # FIXME: now assume the image is square, and split to 2x2 patches
                        # num_patches = h * w, where h = w = sqrt(num_patches)
                        # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                        # we want to first unflatten it to (2, 2, h, w, hidden_size)
                        # rank0_print("At least we are reaching here")
                        # import pdb; pdb.set_trace()
                        if image_idx in video_idx_in_batch:  # video operations
                            # rank0_print("Video")
                            if mm_newline_position == "grid":
                                # Grid-wise
                                image_feature = self.add_token_per_grid(image_feature)
                                if getattr(self.config, "add_faster_video", False):
                                    faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                    # Add a token for each frame
                                    concat_slow_fater_token = []
                                    # import pdb; pdb.set_trace()
                                    for _ in range(image_feature.shape[0]):
                                        if _ % self.config.faster_token_stride == 0:
                                            concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                        else:
                                            concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    # import pdb; pdb.set_trace()
                                    image_feature = torch.cat(concat_slow_fater_token)

                                    # print("!!!!!!!!!!!!")
                            
                                new_image_features.append(image_feature)
                            elif mm_newline_position == "frame":
                                # Frame-wise
                                image_feature = self.add_token_per_frame(image_feature)

                                new_image_features.append(image_feature.flatten(0, 1))
                                
                            elif mm_newline_position == "one_token":
                                # one-token
                                image_feature = image_feature.flatten(0, 1)
                                if 'unpad' in mm_patch_merge_type:
                                    image_feature = torch.cat((
                                        image_feature,
                                        self.model.image_newline[None].to(image_feature.device)
                                    ), dim=0)
                                new_image_features.append(image_feature)      
                            elif mm_newline_position == "no_token":
                                new_image_features.append(image_feature.flatten(0, 1))
                            else:
                                raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                        elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                            # rank0_print("Single-images")
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]

                            if "anyres_max" in image_aspect_ratio:
                                matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                                if matched_anyres_max_num_patches:
                                    max_num_patches = int(matched_anyres_max_num_patches.group(1))

                            if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                                if hasattr(self.get_vision_tower(), "image_size"):
                                    vision_tower_image_size = self.get_vision_tower().image_size
                                else:
                                    raise ValueError("vision_tower_image_size is not found in the vision tower.")
                                try:
                                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                                except Exception as e:
                                    rank0_print(f"Error: {e}")
                                    num_patch_width, num_patch_height = 2, 2
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                image_feature = image_feature.view(2, 2, height, width, -1)

                            if "maxpool2x2" in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = nn.functional.max_pool2d(image_feature, 2)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                                unit = image_feature.shape[2]
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                c, h, w = image_feature.shape
                                times = math.sqrt(h * w / (max_num_patches * unit**2))
                                if times > 1.1:
                                    image_feature = image_feature[None]
                                    image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                                image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            elif "unpad" in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            if "nobase" in mm_patch_merge_type:
                                pass
                            else:
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            new_image_features.append(image_feature)
                        else:  # single image operations
                            image_feature = image_feature[0]
                            if "unpad" in mm_patch_merge_type:
                                image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                            new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                image_features = self.encode_images(und_images)
            rank0_print("image_features: ", image_features.shape)


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # _input_ids = input_ids
        # input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        # labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                # cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds_1)
                new_labels.append(labels[batch_idx])
                # cur_image_idx += 1
                continue
            # 把文本块与图像块拆/嵌/再拼
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            assert cur_image_idx + num_images <= len(image_features), \
                f"image_features 不足：需要 {num_images}，但只剩 {len(image_features) - cur_image_idx}"

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        min_len = min(x.shape[0] for x in new_input_embeds)
        if not min_len == max_len:
            rank0_print("min_len != max_len")
            batch_size = len(new_input_embeds)

            new_input_embeds_padded = []
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            
                cur_len = cur_new_embed.shape[0]
                if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                    new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask[i, -cur_len:] = True
                        position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask[i, :cur_len] = True
                        position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
            new_labels = new_labels_padded
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_labels = torch.stack(new_labels, dim=0)
            if position_ids.dim() == 1:  # [L]
                position_ids = position_ids.unsqueeze(0).expand(attention_mask.shape[0], -1) 
        
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        # add the gen_image_latent query to the input_embed
        image_idx = (input_ids == IMAGE_TOKEN_IDX_LLADA)
        output_indicator = new_labels != IGNORE_INDEX
        gen_img_idx = torch.logical_and(output_indicator, image_idx)

        new_input_embeds = new_input_embeds.clone() 
        new_input_embeds[gen_img_idx] = latent_queries

        labels[image_idx] = -100


        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, target_image_embeds

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
