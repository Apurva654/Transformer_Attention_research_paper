"""
                          Position-wise Feed-Forward Network
                              Paper Section 3.3
 Equation 2:     FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
         This is just two linear layers with a ReLU in between.
         Applied to EACH POSITION INDEPENDENTLY AND IDENTICALLY

"""
import torch
import torch.nn as nn
#---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):  
        # d_model:input/output dimension (512)
        # d_ff:inner (hidden) dimension (2048)
       
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   
        self.linear2 = nn.Linear(d_ff, d_model)  

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
#---------------------------------------------------------------------------------
    """
        Operations: x → Linear(512→2048) → ReLU → Dropout → Linear(2048→512)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand to higher-dimensional space
        x = self.relu(self.linear1(x)) 
        x = self.dropout(x)

        # Project back to d_model
        x = self.linear2(x)          
        return x