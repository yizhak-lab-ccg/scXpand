import numpy as np
import torch

from scxpand.autoencoders.ae_models import AutoencoderModel, ForkAutoencoder
from scxpand.data_util.data_format import DataFormat


# Create a simple DataFormat for testing
data_format = DataFormat(
    n_genes=100, genes_mu=np.zeros(100), genes_sigma=np.ones(100), eps=1e-4, use_log_transform=False, target_sum=1e4
)

# Test AutoencoderModel
model1 = AutoencoderModel(
    data_format=data_format,
    latent_dim=10,
    encoder_hidden_dims=(64,),
    decoder_hidden_dims=(64,),
    classifier_hidden_dims=(16,),
    dropout_rate=0.3,
)
x = torch.randn(5, 100)
latent = model1.encode(x)
output = model1.decode(latent)
print(f"AutoencoderModel decode output type: {type(output)}")
print(f"Output mean shape: {output.mu.shape}")
print(f"Output pi: {output.pi is not None}")
print(f"Output theta: {output.theta is not None}")

# Test ForkAutoencoder
model2 = ForkAutoencoder(
    data_format=data_format,
    latent_dim=10,
    encoder_hidden_dims=(64,),
    decoder_hidden_dims=(64,),
    classifier_hidden_dims=(16,),
    dropout_rate=0.3,
)
output2 = model2.decode(latent)
print(f"ForkAutoencoder decode output type: {type(output2)}")
print("All tests passed!")
