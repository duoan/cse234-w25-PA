import math
from typing import Callable, List, Tuple

import auto_diff as ad
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from torchvision import datasets, transforms

max_len = 28


def _linear(
    x: ad.Node,
    w: ad.Node,
    b: ad.Node,
    batch_size: int,
    seq_length: int,
    out_dim: int,
) -> ad.Node:
    """output = x @ w + b, with bias broadcast to (batch_size, seq_length, out_dim)."""
    xw = ad.matmul(x, w)
    b_broadcast = ad.broadcast(
        b, input_shape=[out_dim], target_shape=[batch_size, seq_length, out_dim]
    )
    return xw + b_broadcast


def _single_head_attention(
    x: ad.Node,
    w_q: ad.Node,
    w_k: ad.Node,
    w_v: ad.Node,
    model_dim: int,
) -> ad.Node:
    """Scaled dot-product self-attention: Softmax(Q K^T / sqrt(d_k)) V."""
    q = ad.matmul(x, w_q)
    k = ad.matmul(x, w_k)
    v = ad.matmul(x, w_v)

    scores = ad.matmul(q, ad.transpose(k, -1, -2))
    scaled = ad.div_by_const(scores, math.sqrt(model_dim))
    attn = ad.softmax(scaled, dim=-1)
    return ad.matmul(attn, v)


def transformer(
    X: ad.Node,
    nodes: List[ad.Node],
    model_dim: int,
    seq_length: int,
    eps,
    batch_size,
    num_classes,
) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, input_dim), denoting the input data.
    nodes: List[ad.Node]
        Weight/bias nodes in the order [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2].
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length
        for classification, in shape (batch_size, num_classes).
    """
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    attn_out = _single_head_attention(X, W_Q, W_K, W_V, model_dim)
    proj = ad.matmul(attn_out, W_O)
    h = ad.layernorm(proj, normalized_shape=[model_dim], eps=eps)

    hidden = _linear(h, W_1, b_1, batch_size, seq_length, model_dim)
    hidden = ad.relu(hidden)
    logits_seq = _linear(hidden, W_2, b_2, batch_size, seq_length, num_classes)

    # Average over the sequence dimension to get (batch_size, num_classes).
    return ad.mean(logits_seq, dim=(1,), keepdim=False)


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Average cross-entropy loss over the batch.

    L = - (1 / batch_size) * sum(y_one_hot * log(softmax(Z)))
    """
    probs = ad.softmax(Z, dim=-1)
    log_probs = ad.log(probs)
    per_sample = ad.sum_op(y_one_hot * log_probs, dim=(1,), keepdim=False)
    total = ad.sum_op(per_sample, dim=(0,), keepdim=False)
    return ad.mul_by_const(total, -1.0 / batch_size)


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> Tuple[List[torch.Tensor], float]:
    """Run one epoch of SGD."""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_loss = 0.0
    seen = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue

        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        results = f_run_model(model_weights, X_batch, y_batch)
        # results = [logits, loss, *grads]
        loss_val = results[1]
        grads = results[2:]

        new_weights: List[torch.Tensor] = []
        for w, g in zip(model_weights, grads):
            # MatMul backprop produces gradients with leading batch dims that
            # are not present on the parameter itself; sum them out.
            while g.dim() > w.dim():
                g = g.sum(dim=0)
            new_weights.append(w - lr * g)
        model_weights = new_weights

        total_loss += float(loss_val) * (end_idx - start_idx)
        seen += end_idx - start_idx

    average_loss = total_loss / max(seen, 1)
    print("Avg_loss:", average_loss)
    return model_weights, average_loss


def train_model():
    """Train a single transformer layer on MNIST."""
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10
    model_dim = 128
    eps = 1e-5

    # Training settings
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # --- Define the forward graph ---
    X = ad.Variable(name="X")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")
    weight_nodes: List[ad.Node] = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_predict: ad.Node = transformer(
        X,
        weight_nodes,
        model_dim=model_dim,
        seq_length=seq_length,
        eps=eps,
        batch_size=batch_size,
        num_classes=num_classes,
    )
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)

    # --- Backward graph ---
    grads: List[ad.Node] = ad.gradients(loss, weight_nodes)

    # --- Evaluators ---
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # --- Load MNIST ---
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_test = test_dataset.targets.numpy()

    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # --- Initialize weights ---
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(model_weights, X_batch, y_batch):
        """Forward + backward for one mini-batch. Returns [logits, loss, *grads]."""
        feed = {
            X: X_batch,
            y_groundtruth: y_batch,
            W_Q: model_weights[0],
            W_K: model_weights[1],
            W_V: model_weights[2],
            W_O: model_weights[3],
            W_1: model_weights[4],
            W_2: model_weights[5],
            b_1: model_weights[6],
            b_2: model_weights[7],
        }
        return evaluator.run(input_values=feed)

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """Forward only; returns predicted class labels for X_val."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        all_logits = []
        for i in range(num_batches):
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run(
                {
                    X: X_batch,
                    W_Q: model_weights[0],
                    W_K: model_weights[1],
                    W_V: model_weights[2],
                    W_O: model_weights[3],
                    W_1: model_weights[4],
                    W_2: model_weights[5],
                    b_1: model_weights[6],
                    b_2: model_weights[7],
                }
            )
            all_logits.append(logits[0])
        concatenated_logits = np.concatenate([l.numpy() for l in all_logits], axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # --- Train ---
    X_train, X_test, y_train, y_test = (
        torch.tensor(X_train),
        torch.tensor(X_test),
        torch.DoubleTensor(y_train),
        torch.DoubleTensor(y_test),
    )

    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val),
        torch.tensor(W_K_val),
        torch.tensor(W_V_val),
        torch.tensor(W_O_val),
        torch.tensor(W_1_val),
        torch.tensor(W_2_val),
        torch.tensor(b_1_val),
        torch.tensor(b_2_val),
    ]

    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label == y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
