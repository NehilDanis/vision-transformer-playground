from network import SelfAttention, self_attention
import torch


# before running this test make sure to set the random seed in the SelfAttention 
# class to a fixed value (e.g., 42) to ensure reproducibility of the results. 
# This is important because the weights of the linear layers are initialized randomly,
# and without a fixed seed, the output may vary between runs, making it difficult to 
# compare against a ground truth result.
def test_self_attention():
    encodings_matrix = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.41, -2.16]])

    self_attention = SelfAttention(embed_dim=2)

    output = self_attention(encodings_matrix)
    assert output.shape == encodings_matrix.shape

    gt_result = torch.tensor([[1.0100, 1.0641],
        [0.2040, 0.7057],
        [3.4989, 2.2427]])
    assert torch.allclose(output, gt_result, atol=1e-4)

