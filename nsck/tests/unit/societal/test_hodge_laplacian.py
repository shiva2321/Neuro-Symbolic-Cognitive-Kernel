import pytest
import numpy as np
from python.core.vsa.hodge_laplacian import HodgeLaplacian

def test_hodge_laplacian_l0():
    # Simple line graph A-B-C
    hl = HodgeLaplacian(["A", "B", "C"])
    
    hl.add_edge("A", "B", 1.0)
    hl.add_edge("B", "C", 1.0)
    
    L0 = hl.compute_L0()
    
    # Degree matrix D:
    # A=1, B=2, C=1
    # Adjacency W:
    # A-B=1, B-A=1, B-C=1, C-B=1
    
    # L0 = D - W
    expected_L0 = np.array([
        [ 1., -1.,  0.],
        [-1.,  2., -1.],
        [ 0., -1.,  1.]
    ], dtype=np.float32)
    
    np.testing.assert_array_almost_equal(L0, expected_L0)

def test_hodge_laplacian_l1():
    # Triangle graph A-B-C
    hl = HodgeLaplacian(["A", "B", "C"])
    
    hl.add_edge("A", "B", 1.0)
    hl.add_edge("B", "C", 1.0)
    hl.add_edge("C", "A", 1.0)
    
    L1 = hl.compute_L1()
    
    assert L1.shape == (3, 3) # m x m where m is edges
    
    # B1^T * B1 should be symmetric and positive semi-definite
    assert np.allclose(L1, L1.T)
    eigenvalues = np.linalg.eigvalsh(L1)
    assert np.all(eigenvalues >= -1e-5) # PSD check

def test_hodge_laplacian_l2():
    # Tetrahedron graph A-B-C-D
    hl = HodgeLaplacian(["A", "B", "C", "D"])
    
    # All 6 edges for a K4
    edges = [("A", "B"), ("A", "C"), ("A", "D"),
             ("B", "C"), ("B", "D"), ("C", "D")]
             
    for u, v in edges:
        hl.add_edge(u, v, 1.0)
        
    L2 = hl.compute_L2()
    
    # 6 edges -> L2 is 6x6
    assert L2.shape == (6, 6)
    
    # K4 has 4 faces (triangles), B2 should be 6x4, L2 = B2 * B2.T
    # PSD check
    assert np.allclose(L2, L2.T)
    eigenvalues = np.linalg.eigvalsh(L2)
    assert np.all(eigenvalues >= -1e-5)
    
    # The rank of L2 for a tetrahedron is 3
    rank = np.linalg.matrix_rank(L2)
    assert rank == 3
