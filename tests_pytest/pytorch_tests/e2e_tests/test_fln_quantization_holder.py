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
import model_compression_toolkit as mct
import torch
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchFLNActivationQuantizationHolder

from model_compression_toolkit.core import CoreConfig
from tests_pytest._fw_tests_common_base.base_fusing_test import build_activation_mp_tpc


def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs

def get_float_model():
    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.sigmoid = torch.nn.Sigmoid()
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(in_features=48, out_features=10)
            self.hswish = torch.nn.Hardswish()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.sigmoid(x)
            x = self.flatten(x)
            x = self.fc(x)
            x = self.hswish(x)
            return x
    return BaseModel()

def test_fln_quantization_holder():

    float_model = get_float_model()
    tpc = build_activation_mp_tpc()

    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_data_gen,
        core_config=CoreConfig(),
        target_platform_capabilities=tpc
    )

    # check conv1
    conv1_activation_holder_quantizer = quantized_model.conv1_activation_holder_quantizer
    assert isinstance(conv1_activation_holder_quantizer, PytorchFLNActivationQuantizationHolder)
    assert conv1_activation_holder_quantizer.quantization_bypass == True

    # check relu
    relu_activation_holder_quantizer = quantized_model.relu_activation_holder_quantizer
    assert isinstance(relu_activation_holder_quantizer, PytorchActivationQuantizationHolder)

    # check conv2
    conv2_activation_holder_quantizer = quantized_model.conv2_activation_holder_quantizer
    assert isinstance(conv2_activation_holder_quantizer, PytorchFLNActivationQuantizationHolder)
    assert conv2_activation_holder_quantizer.quantization_bypass == True

    # check sigmoid
    sigmoid_activation_holder_quantizer = quantized_model.sigmoid_activation_holder_quantizer
    assert isinstance(sigmoid_activation_holder_quantizer, PytorchActivationQuantizationHolder)

    # check fc
    fc_activation_holder_quantizer = quantized_model.fc_activation_holder_quantizer
    assert isinstance(fc_activation_holder_quantizer, PytorchFLNActivationQuantizationHolder)
    assert fc_activation_holder_quantizer.quantization_bypass == True

    # check hswish
    hswish_activation_holder_quantizer = quantized_model.hswish_activation_holder_quantizer
    assert isinstance(hswish_activation_holder_quantizer, PytorchActivationQuantizationHolder)