# Checkpoints

This folder is intentionally empty. Place pretrained weights here before
running `app.py`.

## Required files

```
checkpoints/
  vpt/
    2x.model                     # VPT 2x architecture descriptor
  steve1/
    steve1.weights               # STEVE-1 main weights
    steve1_prior.pt              # STEVE-1 VAE text-to-latent prior
  mineclip/
    attn.pth                     # MineCLIP attention weights
```

## Where to download

- **STEVE-1 (`steve1.weights`, `steve1_prior.pt`)**: original release at
  <https://github.com/Shalev-Lifshitz/STEVE-1> or the HuggingFace mirror
  `CraftJarvis/MineStudio_STEVE-1.official` (used automatically when running
  with `MineStudio` >= 0.0.5).
- **VPT (`2x.model`)**: OpenAI's official VPT release.
- **MineCLIP (`attn.pth`)**: NVIDIA's MineCLIP release.

If you install MineStudio via `pip install MineStudio`, the loader at
`src/mineevolve/executor/steve_loader.py` can fetch these weights from
HuggingFace automatically. In that case this folder may stay empty.
