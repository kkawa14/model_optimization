# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy
from typing import Callable

import torch.nn

from mct_quantizers import PytorchQuantizationWrapper
from mct_quantizers.common.constants import LAYER, WEIGHTS_QUANTIZERS
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.exporter.model_exporter.fw_agonstic.exporter import Exporter

def find_and_assign_metadata_attr(model: torch.nn.Module, attr_name: str = 'metadata'):
    """
    Searches for a given attribute in the model and its submodules.
    If found, assigns the first occurrence to the top-level model under the same attribute name.
    Warn if the attribute is not found or found in multiple places.

    Args:
        model (torch.nn.Module): The model to search.
        attr_name (str): The name of the attribute to look for. Default is 'metadata'.
    """
    found_attrs = []

    def _search(m):
        """Recursively search the model and its submodules for the attribute."""
        if hasattr(m, attr_name):
            found_attrs.append(getattr(m, attr_name))
        for child in m.children():
            _search(child)

    _search(model)

    if not found_attrs:
        # Warn if the attribute was not found anywhere
        Logger.warning(f"Attribute '{attr_name}' not found in the model or its submodules.")
    else:
        setattr(model, attr_name, found_attrs[0])

        if len(found_attrs) > 1:
            # Warn if the attribute was found in multiple places
            Logger.warning(
                f"Attribute '{attr_name}' found in {len(found_attrs)} places. "
                f"Only the first one was assigned to 'model.metadata'.")


def _set_quantized_weights_in_wrapper(layer:PytorchQuantizationWrapper):
    """
       Sets the quantized weights in the provided PytorchQuantizationWrapper layer.
       Replaces the original weights in the layer with the quantized weights.

       Args:
           layer (PytorchQuantizationWrapper): The layer containing quantized weights.

       Raises:
           AssertionError: If the provided layer is not an instance of PytorchQuantizationWrapper.
    """
    assert isinstance(layer, PytorchQuantizationWrapper), f' Expected module {layer} to be PytorchQuantizationWrapper but is of type {type(layer)}'

    # Replace the weights in the layer with quantized weights
    for name in layer.weights_quantizers.keys():
        quantized_weight = torch.nn.Parameter(layer.get_quantized_weights()[name]).detach()
        linear_layer = getattr(layer, LAYER)
        delattr(linear_layer, name)
        setattr(linear_layer, name, torch.nn.Parameter(quantized_weight))

    # Clear the weights quantizers dictionary
    layer.weights_quantizers = {}


class BasePyTorchExporter(Exporter):
    """
    Base PyTorch exporter class.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 is_layer_exportable_fn: Callable,
                 save_model_path: str,
                 repr_dataset: Callable):
        """
        Args:
            model: Model to export.
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            save_model_path: Path to save the exported model.
            repr_dataset: Representative dataset (needed for creating torch script).

        """
        super().__init__(model,
                         is_layer_exportable_fn,
                         save_model_path)

        self.model = copy.deepcopy(self.model)
        self.repr_dataset = repr_dataset

    def _substitute_fully_quantized_model(self):
        """
        Substitution for pytorch "fully-quantized" models. It first uses the weight quantizers
        in PytorchQuantizationWrapper layers to quantize the weights and set them in the layer.
        Then, it replaces all wrapped layers with the layers the wrap.
        """

        # Replace float weight with wrapped quantized weights
        for layer in self.model.modules():
            if isinstance(layer, PytorchQuantizationWrapper):
                _set_quantized_weights_in_wrapper(layer)

        # Replace PytorchQuantizationWrapper layers with their internal layers
        self._replace_wrapped_with_unwrapped()

    def _replace_wrapped_with_unwrapped(self):
        """
        Replaces the PytorchQuantizationWrapper modules in the model with their underlying wrapped modules.
        Iterates through the model's children and replaces the PytorchQuantizationWrapper instances with their
        internal layers.
        """
        for name, module in self.model.named_children():
            if isinstance(module, PytorchQuantizationWrapper):
                setattr(self.model, name, module.layer)
