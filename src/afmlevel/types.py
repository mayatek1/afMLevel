"""
Type definitions for U-Net model configuration.

This module provides static typing helpers for configuration dictionaries used
when constructing and loading U-Net models. The types defined here improve
documentation clarity and enable static type checking without affecting runtime
behaviour.
"""

from typing import TypedDict


class UNetConfig(TypedDict, total=False):
    """
    Configuration dictionary for U-Net model construction.

    This TypedDict defines the optional keyword arguments accepted by the
    U-Net model constructor. All keys are optional, as the U-Net implementation
    provides sensible default values for each parameter.

    This type is intended for use with ``**config`` expansion when constructing
    U-Net instances, enabling static checking of configuration keys and value
    types without constraining runtime flexibility.

    Fields
    ------
    filtersize : int
        Kernel size used in convolutional layers. Must be an odd integer.
    leakyrelu : bool
        Whether to use LeakyReLU activations instead of ReLU.
    dropoutprob : float
        Dropout probability applied after convolutional layers.

    Notes
    -----
    - This type is used only for typing and documentation purposes.
    - Missing keys imply that the U-Net default parameter values will be used.
    """

    filtersize: int
    leakyrelu: bool
    dropoutprob: float
