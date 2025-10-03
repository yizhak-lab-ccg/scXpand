import pytest
import torch

from scxpand.data_util.data_format import DataFormat
from scxpand.mlp.mlp_model import FC_Net, MLPModel, ModelOutput
from scxpand.mlp.mlp_params import MLPParam


@pytest.fixture
def mock_data_format() -> DataFormat:
    """Create a mock DataFormat for testing."""
    data_format = DataFormat()
    data_format.n_genes = 10
    data_format.aux_categorical_types = ["tissue_type", "imputed_labels"]
    return data_format


@pytest.fixture
def mock_params() -> MLPParam:
    """Create mock parameters for testing."""
    return MLPParam(
        n_epochs=10,
        layer_units=[32, 16],
        dropout_rate=0.1,
    )


def test_fc_net_forward():
    """Test FC_Net forward pass."""
    # Create a network
    net = FC_Net(in_dim=15, out_dim=1, hid_layers=(32, 16), dropout_rate=0.1)

    # Create a mock input
    batch_size = 5
    x = torch.randn(batch_size, 15)

    # Run forward pass
    output = net(x)

    # Check output shape
    assert output.shape == (batch_size, 1)


def test_nn_model_initialization(mock_data_format, mock_params):
    """Test NNModel initialization."""
    # Create model
    model = MLPModel(prm=mock_params, data_format=mock_data_format)

    # Check attributes
    assert model.n_input_features == 10
    assert model.data_format.n_genes == 10
    assert isinstance(model.fc_net, FC_Net)

    # Input dimension should be the sum of gene count
    assert model.fc_net.in_dim == 10


def test_nn_model_forward(mock_data_format, mock_params):
    """Test NNModel forward pass."""
    # Create model
    model = MLPModel(prm=mock_params, data_format=mock_data_format)

    # Create mock batch
    batch_size = 3
    input_tensor = torch.randn(batch_size, 10)

    # Run forward pass
    output = model(input_tensor)

    # Check output is ModelOutput instance
    assert isinstance(output, ModelOutput)

    # Check main_logit shape - should be [batch_size]
    assert output.main_logit.shape == (batch_size,)

    # Check categorical_logits is None (since no aux_categorical_mappings in mock)
    assert output.categorical_logits is None
