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

import torch.nn

from model_compression_toolkit.target_platform_capabilities import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from tests_pytest._fw_tests_common_base.base_tpc_attach2fw_test import BaseTpcAttach2FrameworkTest


class TestAttachTpc2Pytorch(BaseTpcAttach2FrameworkTest):

    attach2fw_class = AttachTpcToPytorch

    def test_attach2fw_init(self):
        super().test_attach2fw_init()

    def test_attach2fw_attach_without_attributes(self):
        super().test_attach2fw_attach_without_attributes()

    def test_attach2fw_attach_linear_op_with_attributes(self):
        super().test_attach2fw_attach_linear_op_with_attributes()

    def test_attach2fw_attach_to_default_config(self):
        super().test_attach2fw_attach_to_default_config()

    def test_not_existing_opset_with_layers_to_attach(self):
        super().test_not_existing_opset_with_layers_to_attach()

    def test_attach2pytorch_attach_with_custom_opset(self):
        self._test_attach2fw_attach_with_custom_opset([torch.nn.Identity],
                                                      LayerFilterParams(torch.nn.Conv2d, stride=2),
                                                      "CustomAttr")

    def test_attach2pytorch_prioritize_custom_opset(self):
        self._test_attach2fw_prioritize_custom_opset(torch.nn.Conv2d)
