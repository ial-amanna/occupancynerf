"""
This code contains the NGP radiance field class along with the new class to get the occupancy conditioned on radius called NewESDF

Parts of the program are taken from the Instant NGP nerfacc repository

"""

from typing import Callable, List, Union
from collections import OrderedDict
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

#check installation of tinycudann
try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

#Force grid encoder to use float32 (optimization)
class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


# fix query density to unisphere if unbounded
def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    eps: float = 1e-6,
    derivative: bool = False,
):
    #splits the tensor into two 3d tensors
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x


# Instant NGP Class model
class NGPradianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded

        self.geo_feat_dim = geo_feat_dim
        per_level_scale = 1.4472692012786865

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                    ],
                },
            )

        self.mlp_base_1 = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )
        self.mlp_base_2 = tcnn.Network(
            n_input_dims=self.mlp_base_1.n_output_dims,
            n_output_dims= 1+self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )


        self.mlp_base = torch.nn.Sequential(self.mlp_base_1, self.mlp_base_2)



        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    (
                        self.direction_encoding.n_output_dims
                        if self.use_viewdirs
                        else 0
                    )
                    + self.geo_feat_dim
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def query_density(self, x, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.view(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .view(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
            density, embedding = self.query_density(positions, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density

# New class for the ESDF
class NewESDF(torch.nn.Module):
    """Just for Occupancy conditioned on radius from Density"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        new_tensor: torch.Tensor, #torch.full((1, 5), 1.0, dtype=torch.float32)
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        radius_dim: int = 5,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.radius_dim = radius_dim

        self.new_tensor = new_tensor
        self.new_tensor = self.new_tensor.reshape((1,5))

        self.geo_feat_dim = geo_feat_dim
        per_level_scale = 1.4472692012786865

        # MLP Layers, we add one more layer. On analysis, instead of having NetworkwithEncoding together, separating them, we got better results.
        # This is due to the encoding when the params are kept separate, don't interfere with the network params, which is of importance for the second phase of training.

        self.mlp_base_1 = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )
        self.mlp_base_2 = tcnn.Network(
            n_input_dims=self.mlp_base_1.n_output_dims,
            n_output_dims= 1+self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        #sigmoid for last layer, adding the dimension for radius
        self.mlp_base_3 = tcnn.Network(
            n_input_dims= 1+self.geo_feat_dim + self.radius_dim,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid", 
                "n_neurons": 1+self.geo_feat_dim,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_base = torch.nn.Sequential(self.mlp_base_1, self.mlp_base_2)



    def query_density(self, x, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        temp_tensor = self.new_tensor.expand(x.shape[0], 5)
        x = (
            self.mlp_base_3(torch.cat((self.mlp_base(x.view(-1, self.num_dim)), temp_tensor), dim=1))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density


    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
            density, embedding = self.query_density(positions, return_feat=True)
        return density