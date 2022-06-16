import dataclasses
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from dpl.emoca.models import ResnetEncoder, Generator


@dataclasses.dataclass
class FlameDims:
    shape: int = 100
    tex: int = 50
    exp: int = 50
    pose: int = 6
    cam: int = 3
    light: int = 27

    @classmethod
    def default(cls) -> 'FlameDims':
        return cls()

    def get_ordered_lengths(self, order: Optional[List[str]] = None) -> List[int]:
        if order is None:
            return [
                self.shape,
                self.tex,
                self.exp,
                self.pose,
                self.cam,
                self.light,
            ]
        return [getattr(self, key) for key in order]

    def total(self) -> int:
        return sum(self.get_ordered_lengths())


class EmocaInferenceWrapper(nn.Module):
    def __init__(
        self,
        flame_dims: FlameDims = FlameDims.default(),
        n_detail: int = 128,
        detail_conditioning: Optional[List[str]] = None,
    ):
        super().__init__()

        self.dims = flame_dims
        self.n_detail = n_detail

        self.flame_order = ['shape', 'tex', 'exp', 'pose' , 'cam', 'light']

        if detail_conditioning is None:
            # Only needed for detail tecture reconstruction.
            # See stage 3 in decode() of the original DecaModule.
            self.detail_conditioning = ['jawpose', 'expression', 'detail']
        else:
            self.detail_conditioning = detail_conditioning

        self._create_model()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward encoding pass of the model.
        Takes a batch of images and returns the corresponding latent codes for each image.
        
        Parameters
        ----------
        images: Batch of images to encode, shape (B, [K,] 3, H, W). 

        Returns
        -------
        Dict containing predicted FLAME parameters.
        """
        codedict = {}

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        nc, h, w = images.shape[-3 :]
        images = images.view(-1, nc, h, w)
        codedict['images'] = images

        # 1) COARSE STAGE
        # forward pass of the coarse encoder
        params = self._encode_flame(images)
        codedict.update(params)

        # 2) DETAIL STAGE
        all_detailcode = self.E_detail(images)
        # identity-based detail code
        detailcode = all_detailcode[:, : self.n_detail]
        codedict['detailcode'] = detailcode

        return codedict

    def decompose_code(self, code: torch.Tensor) -> Dict[str, torch.Tensor]:
        params = {}
        lengths = self.dims.get_ordered_lengths(self.flame_order)

        start = 0
        for key, length in zip(self.flame_order, lengths):
            params[key] = code[:, start: start + length]
            start += length

        params['light'] = params['light'].reshape(code.shape[0], 9, 3)
        return params

    def _compute_condition_dim(self):
        n_cond = 0
        if 'globalpose' in self.detail_conditioning:
            n_cond += 3
        if 'jawpose' in self.detail_conditioning:
            n_cond += 3
        if 'identity' in self.detail_conditioning:
            n_cond += self.dims.shape
        if 'expression' in self.detail_conditioning:
            n_cond += self.dims.exp

        return n_cond

    def _create_model(self) -> None:
        self.E_flame = ResnetEncoder(outsize=self.dims.total())
        self.E_detail = ResnetEncoder(outsize=self.n_detail)
        n_cond = self._compute_condition_dim()
        self.D_detail = Generator(
            latent_dim=self.n_detail + n_cond,
            out_channels=1,
            out_scale=0.01,
            sample_mode='bilinear',
        )
        self.E_expression = ResnetEncoder(self.dims.exp)

    def _encode_flame(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        deca_code = self.E_flame(images)
        expcode = self.E_expression(images)
        params = self.decompose_code(deca_code)
        params['exp'] = expcode
        return params
