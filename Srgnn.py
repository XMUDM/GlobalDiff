
import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )
        self.linear_edge_out = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = (
            torch.matmul(A[:, :, : A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        )
        input_out = (
            torch.matmul(
                A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden)
            )
            + self.b_ioh
        )
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SRGNN(nn.Module):
    r"""SRGNN regards the conversation history as a directed graph.
    In addition to considering the connection between the item and the adjacent item,
    it also considers the connection with other interactive items.

    Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connection matrix A

    Outgoing edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     1     0     0
         2    0     0    1/2   1/2
         3    0     1     0     0
         4    0     0     0     0
        === ===== ===== ===== =====

    Incoming edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     0     0     0
         2   1/2    0    1/2    0
         3    0     1     0     0
         4    0     1     0     0
        === ===== ===== ===== =====
    """

    def __init__(self, device, item_num):
        super(SRGNN, self).__init__()

        # load parameters info
        self.embedding_size = 256
        self.step = 1
        self.device = device
        self.loss_type = "CE"
        self.n_items = item_num
        # define layers and loss
        # item embedding
        self.item_embedding = nn.Embedding(
            self.n_items+2, self.embedding_size, padding_idx=0
        )
        # define layers and loss
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.embedding_size * 2, self.embedding_size, bias=True
        )
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self._reset_parameters()


    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        # 生成一个布尔掩码矩阵，标记 item_seq 中非零的部分
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        # 对每个会话序列 u_input，提取其中的唯一项目节点（node），并将其补零到最大长度 max_n_node，将得到的节点列表添加到 items。
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            # 遍历会话中的每对相邻项目（u_input[i] 和 u_input[i + 1]）构建邻接矩阵 u_A。u 和 v 是当前和下一个节点的索引。
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            # u_sum_in = np.sum(u_A, 0) 计算每列的和，即每个节点的入度之和（即有多少节点指向该节点）。
            u_sum_in = np.sum(u_A, 0)
            # 某些节点可能没有入度或出度（即 u_sum_in 或 u_sum_out 中有为零的情况），这会导致后续除法操作报错。
            # 将 u_sum_in 和 u_sum_out 中的零值替换为 1，以避免后续的除法中出现零除错误。
            u_sum_in[np.where(u_sum_in == 0)] = 1
            # 对 u_A 的每列进行归一化，得到入度归一化的邻接矩阵 u_A_in。
            u_A_in = np.divide(u_A, u_sum_in)
            # u_sum_out = np.sum(u_A, 1) 计算每行的和，即每个节点的出度之和（即该节点指向了多少节点）。
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            # 对 u_A 的每行进行归一化（在转置后按列进行归一化），得到出度归一化的邻接矩阵 u_A_out
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            #  将入度和出度归一化矩阵连接起来，形成一个双向邻接矩阵。
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    def forward(self, item_seq, item_seq_len):
        item_seq = item_seq[:,:-1].clone()
        item_seq_len =item_seq_len-1
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        hidden = self.item_embedding(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.embedding_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return seq_output



    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        # 将gather_index重塑为形状为(-1, 1, 1)
        gather_index = gather_index.view(-1, 1, 1)
        # 扩展gather_index的形状以匹配output的最后一个维度
        gather_index = gather_index.expand(-1, -1, output.shape[-1])
        # 使用gather函数从output中收集特定位置的向量
        output_tensor = output.gather(dim=1, index=gather_index)
        # 移除output_tensor的第二个维度
        return output_tensor.squeeze(1)

    def calculate_loss(self, item_seq,target,negative_target):
        item_seq_len = (item_seq != 0).sum(dim=1)
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = target



        if self.loss_type == "BPR":
            neg_items = negative_target
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)

            test_item_emb = self.item_embedding.weight
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            # neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]

            # 使用 einsum 进行相似度计算，并在 K 维度上求和
            neg_score = torch.einsum('bd,bkd->bk', seq_output, neg_items_emb)  # (B, K)
            # 对 K 个负样本的相似度求平均，得到最终的 neg_score (B)
            neg_score = neg_score.mean(dim=-1)  # (B)

            # pos_score = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # neg_score = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss



    def predict(self, item_seq,target):
        item_seq_len = (item_seq != 0).sum(dim=1)
        test_item = target
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, item_seq):
        item_seq_len = (item_seq != 0).sum(dim=1)
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        # 最后一个是mask token 我不要
        scores = scores[:, :-1]
        return scores
