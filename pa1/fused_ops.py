from typing import List

import auto_diff as ad
import torch
from auto_diff import Node, Op


class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self,
        node_A: Node,
        node_B: Node,
        normalized_shape: List[int],
        eps: float = 1e-5,
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(
        self,
        node: Node,
        input_values: List[torch.Tensor],
    ) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        y = torch.matmul(input_values[0], input_values[1])
        return torch.layer_norm(
            y,
            normalized_shape=node.attrs["normalized_shape"],
            eps=node.attrs["eps"],
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A = node.inputs[0]
        B = node.inputs[1]
        eps = node.attrs["eps"]
        normalized_shape = node.attrs["normalized_shape"]

        # Reconstruct the intermediate MatMul node symbolically for the backward path
        X = ad.matmul(A, B)

        # Calculate LayerNorm gradient mathematically using your LayerNormOp pattern
        dims = tuple(range(-len(normalized_shape), 0))
        n = 1
        for s in normalized_shape:
            n *= s

        x_mean = ad.div_by_const(ad.sum_op(X, dim=dims, keepdim=True), n)
        x_centered = ad.sub(X, x_mean)
        var = ad.div_by_const(
            ad.sum_op(ad.mul(x_centered, x_centered), dim=dims, keepdim=True), n
        )
        std = ad.sqrt(ad.add_by_const(var, eps))

        g_mean = ad.div_by_const(ad.sum_op(output_grad, dim=dims, keepdim=True), n)
        # y is the result of LayerNorm(X), which is exactly 'node' itself
        gy_mean = ad.div_by_const(
            ad.sum_op(ad.mul(output_grad, node), dim=dims, keepdim=True), n
        )

        # Gradient of LayerNorm with respect to intermediate matmul result X
        dX = ad.div(ad.sub(ad.sub(output_grad, g_mean), ad.mul(node, gy_mean)), std)

        # Backpropagate dX through the MatMul operation to inputs A and B
        dA = ad.matmul(dX, ad.transpose(B, -1, -2))
        dB = ad.matmul(ad.transpose(A, -1, -2), dX)

        return [dA, dB]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(self, node_A: Node, node_B: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"dim": dim},
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        # 1. Compute Matrix Multiplication: X = A @ B
        matmul_res = torch.matmul(input_values[0], input_values[1])
        # 2. Compute Softmax: Y = Softmax(X)
        return torch.softmax(matmul_res, dim=node.attrs["dim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        A = node.inputs[0]
        B = node.inputs[1]
        dim = node.attrs["dim"]

        # Softmax gradient vector-Jacobian product: dX = Y * (dL/dY - sum(dL/dY * Y, dim, keepdim=True))
        # Note: 'node' holds the forward evaluation result of Softmax(A @ B)
        dot = ad.sum_op(ad.mul(output_grad, node), dim=dim, keepdim=True)
        dX = ad.mul(node, ad.sub(output_grad, dot))

        # Backpropagate dX through the MatMul operation to inputs A and B
        dA = ad.matmul(dX, ad.transpose(B, -1, -2))
        dB = ad.matmul(ad.transpose(A, -1, -2), dX)

        return [dA, dB]


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
