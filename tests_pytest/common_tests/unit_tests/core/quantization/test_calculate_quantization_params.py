import pytest

from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from tests_pytest._test_util.graph_builder_utils import build_node


TEST_KERNEL = 'kernel'
TEST_BIAS = 'bias'

### dummy layer classes
class Conv2D:
    pass
class InputLayer:
    pass
class Add:
    pass
class BatchNormalization:
    pass
class ReLU:
    pass
class Flatten:
    pass
class Dense:
    pass

from unittest.mock import Mock
class DummyLayer:
    """ Only needed for repr(node) to work. """
    pass

@pytest.fixture
def fw_impl_mock():
    """
    Fixture to create a fake framework implementation with a mocked model_builder.
    """
    fw_impl = Mock()
    fw_impl.model_builder.return_value = (Mock(), None)
    return fw_impl


### test model
def get_test_graph():
    n1 = build_node('input', layer_class=InputLayer)
    conv1 = build_node('conv1', layer_class=Conv2D, canonical_weights={TEST_KERNEL: [1,2], TEST_BIAS: [3,4]})
    conv2 = build_node('conv2', layer_class=Conv2D)
    conv3 = build_node('conv3', layer_class=Conv2D)
    sigmoid = build_node('sigmoid', layer_class=Dense)
    relu = build_node('relu1', layer_class=ReLU, canonical_weights={TEST_KERNEL: [1,2], TEST_BIAS: [3,4]})

    graph = Graph('g', input_nodes=[n1],
                  nodes=[conv1, conv2, conv3, sigmoid, relu],
                  output_nodes=[relu],
                  edge_list=[Edge(n1, conv1, 0, 0),
                             Edge(conv1, conv2, 0, 0),
                             Edge(conv2, sigmoid, 0, 0),
                             Edge(sigmoid, conv3, 0, 0),
                             Edge(conv3, relu, 0, 0),
                             ]
                  )


    return graph


class TestCalculateQuantizationParams:
    class AWLayer:
        pass

    class ALayer:
        pass

    class VALayer:
        pass

    kernel_attr = 'im_kernel'

    @pytest.fixture(autouse=True)
    def fw_info_mock(self, fw_info_mock):
        DEFAULT_KERNEL_ATTRIBUTES = [None]
        self.fw_info_mock = fw_info_mock
        self.fw_info_mock.get_kernel_op_attributes = \
            lambda nt: [self.kernel_attr] if nt == self.AWLayer else DEFAULT_KERNEL_ATTRIBUTES


    # test case for test_calculate_quantization_params
    test_input_0 = (None, None)
    test_expected_0 = ("The filters cannot be None.", None)

    @pytest.mark.parametrize(("fw_impl_mock", "inputs", "expected"), [
        (fw_impl_mock, test_input_0, test_expected_0),
    ])
    def test_calculate_quantization_params(self, fw_impl_mock, inputs, expected):
        graph = get_test_graph()
        print()
        print(graph)
        print(graph.nodes)
        graph.implementation = fw_impl_mock
        graph.fw_info = self.fw_info_mock
        calculate_quantization_params(graph, fw_impl_mock, None)
        pass

