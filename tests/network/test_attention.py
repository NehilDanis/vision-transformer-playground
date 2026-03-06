from operator import gt

from network import SelfAttention, Attention, MultiHeadAttention
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

# before running this test make sure to set the random seed in the SelfAttention 
# class to a fixed value (e.g., 42) to ensure reproducibility of the results. 
# This is important because the weights of the linear layers are initialized randomly,
# and without a fixed seed, the output may vary between runs, making it difficult to 
# compare against a ground truth result.
def test_masked_self_attention():
    encodings_matrix = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.41, -2.16]])

    self_attention = SelfAttention(embed_dim=2)

    num_tokens = encodings_matrix.size(0)
    mask = torch.tril(torch.ones(num_tokens, num_tokens)) == 0

    output = self_attention(encodings_matrix, mask=mask)
    assert output.shape == encodings_matrix.shape

    gt_result = torch.tensor([[ 0.6038,  0.7434],
        [-0.0062,  0.6072],
        [ 3.4989,  2.2427]])
    assert torch.allclose(output, gt_result, atol=1e-4)


def test_encoder_decoder_attention():
    ## create matrices of token encodings...
    encodings_for_q = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_k = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_v = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    attention = Attention(embed_dim=2)

    output = attention(encodings_for_q, encodings_for_k, encodings_for_v)
    assert output.shape == encodings_for_q.shape

    gt_result = torch.tensor([[1.0100, 1.0641],
        [0.2040, 0.7057],
        [3.4989, 2.2427]])
    assert torch.allclose(output, gt_result, atol=1e-4)

def test_multi_head_attention():
    ## create matrices of token encodings...
    encodings_for_q = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_k = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_v = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    multi_head_attention = MultiHeadAttention(embed_dim=2, num_heads=1)

    output = multi_head_attention(encodings_for_q, encodings_for_k, encodings_for_v)
    gt_result = torch.tensor([[1.0100, 1.0641],
        [0.2040, 0.7057],
        [3.4989, 2.2427]])
    assert torch.allclose(output, gt_result, atol=1e-4)

    # test with multiple heads
    multi_head_attention = MultiHeadAttention(embed_dim=2, num_heads=2)
    output = multi_head_attention(encodings_for_q, encodings_for_k, encodings_for_v)
    assert output.shape == (3, 4)  # since we have 2 heads  
   
    gt_result_multi_head = torch.tensor([[1.0100, 1.0641, 1.0100, 1.0641],
        [0.2040, 0.7057, 0.2040, 0.7057],
        [3.4989, 2.2427, 3.4989, 2.2427]])
    assert torch.allclose(output, gt_result_multi_head, atol=1e-4)
