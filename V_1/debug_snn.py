from ncgn.brain import Brain
import torch

def debug_snn():
    b = Brain(use_embeddings=False, use_mock_reasoner=True)
    device = b.vsa.device
    
    # 1. Random Input A
    input_a = torch.randn(1, 1, 28, 28).to(device)
    rate_a = b.snn(input_a)
    
    # 2. Random Input B
    input_b = torch.randn(1, 1, 28, 28).to(device)
    rate_b = b.snn(input_b)
    
    # Check difference
    diff = torch.norm(rate_a - rate_b)
    print(f"SNN Rate Vector Diff: {diff.item()}")
    
    print(f"Rate A Mean: {rate_a.mean().item()}")
    print(f"Rate B Mean: {rate_b.mean().item()}")
    
    # Check Bridge
    hv_a = b.bridge(rate_a)[0]
    hv_b = b.bridge(rate_b)[0]
    
    sim = b.vsa.similarity(hv_a, hv_b)
    print(f"VSA Similarity between Random A and B: {sim}")

if __name__ == "__main__":
    debug_snn()
