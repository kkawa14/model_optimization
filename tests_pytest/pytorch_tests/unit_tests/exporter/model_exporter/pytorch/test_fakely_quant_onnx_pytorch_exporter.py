# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import pytest
from unittest.mock import Mock
from mct_quantizers import pytorch_quantizers
from mct_quantizers.pytorch.preserving_activation_quantization_holder import PytorchPreservingActivationQuantizationHolder
from mct_quantizers.pytorch.fln_activation_quantization_holder import PytorchFLNActivationQuantizationHolder
from mct_quantizers.pytorch.activation_quantization_holder import PytorchActivationQuantizationHolder
from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_onnx_pytorch_exporter import FakelyQuantONNXPyTorchExporter

     
class DummyModel:
    def __init__(self):
        activation_holder_quantizer = Mock(spec=pytorch_quantizers.BasePyTorchInferableQuantizer)
        quantization_bypass = True
        self.preserve = PytorchPreservingActivationQuantizationHolder(activation_holder_quantizer, quantization_bypass)
        self.fln = PytorchFLNActivationQuantizationHolder(activation_holder_quantizer, quantization_bypass)
        self.act = PytorchActivationQuantizationHolder(activation_holder_quantizer)

    def named_modules(self):
        return [
            ('preserving_holder', self.preserve),
            ('FLN_quant_holder', self.fln),
            ('act_quant_holder', self.act),
        ]

def test_enable_onnx_custom_ops_export_changes_quantization_bypass():
    """
    Testing the exporter.When PytorchPreservingActivationQuantizationHolder or 
    PytorchFLNActivationQuantizationHolder, make sure that quantization_bypass is converted to False.
    """
    exporter = FakelyQuantONNXPyTorchExporter(
        model=DummyModel(),
        is_layer_exportable_fn=lambda x: True,
        save_model_path="dummy_path",
        repr_dataset=lambda: iter([[]]),
        onnx_opset_version=13,
        use_onnx_custom_quantizer_ops=True
    )

    assert exporter.model.preserve.quantization_bypass is True
    assert exporter.model.fln.quantization_bypass is True

    exporter._enable_onnx_custom_ops_export()

    for n, m in exporter.model.named_modules():
        print("\n n", n)
        print("m", m)
        print("type(m)", type(m))
        print("m", vars(m))

        if n == 'preserving_holder' or n == 'FLN_quant_holder':
            assert m.quantization_bypass is False
            
    assert exporter.model.preserve.quantization_bypass is False
    assert exporter.model.fln.quantization_bypass is False
   
