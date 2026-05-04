import torch
from torch import nn
import torch.nn.functional as F

from models.base import make_contiguous
from models.wan.model import sinusoidal_embedding_1d
from models.wan.wan import FinalLayer, TransformerLayer, WanPipeline, vae_encode
from utils.common import AUTOCAST_DTYPE


DOMAIN_LABELS = {
    'parallel': 0,
    'stereo4d': 0,
    'stereo_4d': 0,
    'converged': 1,
    'converge': 1,
    '3dmovie': 1,
    '3d_movie': 1,
}


def normalize_domain_label(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in DOMAIN_LABELS:
            raise ValueError(f'Unknown stereo domain label {value!r}')
        return DOMAIN_LABELS[key]
    return int(value)


class WanStereoPipeline(WanPipeline):
    """StereoPilot-lite training pipeline for Wan T2V backbones.

    This is a direct left-latent -> right-latent regression prototype. It uses
    diffusion-pipe's existing ``control_path`` dataset mechanism, where the
    target directory contains right-eye videos and control_path contains
    matching left-eye videos with the same filenames.
    """

    name = 'wan_stereo'

    def __init__(self, config):
        super().__init__(config)
        if self.model_type != 't2v':
            raise ValueError('wan_stereo currently expects a Wan T2V checkpoint; start with Wan2.1-T2V-1.3B.')
        self.name = 'wan_stereo'

    def load_diffusion_model(self):
        super().load_diffusion_model()
        dtype = self.model_config['dtype']
        dim = self.transformer.dim

        self.transformer.parall_embedding = nn.Parameter(torch.zeros(6, dim, dtype=dtype))
        self.transformer.converge_embedding = nn.Parameter(torch.zeros(6, dim, dtype=dtype))
        self.transformer.parall_embedding.original_name = 'parall_embedding'
        self.transformer.converge_embedding.original_name = 'converge_embedding'

    def configure_adapter(self, adapter_config):
        super().configure_adapter(adapter_config)
        for name in ('parall_embedding', 'converge_embedding'):
            p = getattr(self.transformer, name)
            p.requires_grad_(True)
            p.data = p.data.to(adapter_config['dtype'])
            p.original_name = name

    def model_specific_dataset_config_validation(self, dataset_config):
        for directory in dataset_config['directory']:
            if 'control_path' not in directory:
                raise ValueError(
                    "wan_stereo requires paired left/right data. Put right-eye videos "
                    "in directory.path and matching left-eye videos in directory.control_path."
                )

    def get_call_vae_fn(self, vae_and_clip):
        def fn(tensor, control_tensor):
            vae = vae_and_clip.vae
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            control_tensor = control_tensor.to(p.device, p.dtype)

            if tensor.ndim != 5 or control_tensor.ndim != 5:
                raise AssertionError(
                    f'wan_stereo must train on videos, got target={tensor.shape}, control={control_tensor.shape}'
                )
            if tensor.shape != control_tensor.shape:
                raise AssertionError(
                    f'left/right video tensors must have matching shapes, got target={tensor.shape}, control={control_tensor.shape}'
                )

            return {
                'latents': vae_encode(tensor, self.vae),
                'stereo_condition': vae_encode(control_tensor, self.vae),
            }
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        target_latents = inputs['latents'].float()
        stereo_condition = inputs['stereo_condition'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            text_embeddings_or_ids = inputs['text_embeddings']
            seq_lens_or_text_mask = inputs['seq_lens']
        else:
            text_embeddings_or_ids, seq_lens_or_text_mask = self.text_encoder.tokenizer(
                inputs['caption'], return_mask=True, add_special_tokens=True
            )

        bs, channels, num_frames, h, w = target_latents.shape
        if stereo_condition.shape != target_latents.shape:
            raise AssertionError(
                f'stereo_condition and target latents must match, got {stereo_condition.shape} vs {target_latents.shape}'
            )

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')
            mask = mask.unsqueeze(2)

        domain_label = inputs.get('domain_label', self.model_config.get('domain_label', 1))
        if torch.is_tensor(domain_label):
            domain_label = domain_label.to(device=stereo_condition.device, dtype=torch.long)
        elif isinstance(domain_label, list):
            domain_label = torch.tensor(domain_label, device=stereo_condition.device, dtype=torch.long)
        else:
            domain_label = torch.full(
                (bs,),
                normalize_domain_label(domain_label),
                device=stereo_condition.device,
                dtype=torch.long,
            )

        if domain_label.ndim == 0:
            domain_label = domain_label.expand(bs)
        domain_label = domain_label.clamp(0, 1)

        t_value = float(self.model_config.get('stereo_timestep', 1.0))
        t = torch.full((bs,), t_value, device=stereo_condition.device, dtype=torch.float32)

        return (
            (stereo_condition, t, text_embeddings_or_ids, seq_lens_or_text_mask, domain_label),
            (target_latents, mask),
        )

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder.model
        layers = [StereoInitialLayer(transformer, text_encoder)]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers


class StereoInitialLayer(nn.Module):
    def __init__(self, model, text_encoder):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.text_embedding = model.text_embedding
        self.time_projection = model.time_projection
        self.parall_embedding = model.parall_embedding
        self.converge_embedding = model.converge_embedding
        self.text_encoder = text_encoder
        self.freqs = model.freqs
        self.freq_dim = model.freq_dim
        self.dim = model.dim
        self.text_len = model.text_len

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_tensor(item) and torch.is_floating_point(item):
                item.requires_grad_(True)

        x, t, text_embeddings_or_ids, seq_lens_or_text_mask, domain_label = inputs

        if self.text_encoder is not None:
            assert not torch.is_floating_point(text_embeddings_or_ids)
            with torch.no_grad():
                context = self.text_encoder(text_embeddings_or_ids, seq_lens_or_text_mask)
            context.requires_grad_(True)
            text_seq_lens = seq_lens_or_text_mask.gt(0).sum(dim=1).long()
        else:
            context = text_embeddings_or_ids
            text_seq_lens = seq_lens_or_text_mask

        context = [emb[:length] for emb, length in zip(context, text_seq_lens)]

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        time_embed_seq_len = seq_len
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            time_embed_seq_len = 1
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, time_embed_seq_len)).to(x.device, torch.float32)
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        domain_table = torch.stack([self.parall_embedding, self.converge_embedding], dim=0)
        domain_label = domain_label.to(device=x.device, dtype=torch.long).clamp(0, 1)
        domain_emb = domain_table[domain_label].unsqueeze(1)
        e0 = e0 + domain_emb.to(e0.dtype)

        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        seq_lens = seq_lens.to(x.device)
        grid_sizes = grid_sizes.to(x.device)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, self.freqs, context)
