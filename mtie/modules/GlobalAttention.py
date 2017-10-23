"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math

_INF = float('inf')

class DoubleGlobalAttention(nn.Module):
    def __init__(self, dim):
        super(DoubleGlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x 2*dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).view(input.size(0), -1, 2)  # batch x dim x 2

        # Get attention
        attn = torch.bmm(context, targetT)  # batch x sourceL x 2
        attn = attn.permute(0, 2, 1).contiguous().view(-1, context.size(1)) # 2*batch x sourceL
        if self.mask is not None:
            if self.mask.size() != attn.size():
                if len(self.mask.size()) == 2:
                    mask3 = self.mask.unsqueeze(1).repeat(1, 2, 1)
                    self.mask = mask3.view(-1, self.mask.size(1))
                else:
                    mask4 = self.mask.unsqueeze(2).repeat(1, 1, 2, 1)
                    self.mask = mask4.view(self.mask.size(0), -1, self.mask.size(2))
            attn.data.masked_fill_(self.mask, -_INF)
        attn = self.sm(attn)
        attn3 = attn.view(-1, 2, attn.size(1))  # batch x 2 x sourceL

        weightedContext = torch.bmm(attn3, context).view(attn3.size(0), -1)  # batch x 2*dim
        contextCombined = torch.cat((weightedContext, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        attn1, attn2 = attn3.split(1, dim=1)

        return contextOutput, attn1.squeeze(1), attn2.squeeze(1)


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -_INF)
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn
