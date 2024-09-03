import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel
from models.hill.coding_tree import get_tree_data
from models.hill.hill import HRLEncoder, GTData


def debug(message):
    print("DEBUG", "=" * 30)
    print(message)
    print("DEBUG", "=" * 30)


class BertPoolingLayer(nn.Module):
    def __init__(self, avg='cls'):
        super(BertPoolingLayer, self).__init__()
        self.avg = avg

    def forward(self, x):
        if self.avg == 'cls':
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return x


class NTXent(nn.Module):

    def __init__(self, config, tau=1., transform=True):
        super(NTXent, self).__init__()
        self.tau = tau
        self.transform = transform
        self.norm = 1.
        if transform:
            self.transform = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            self.transform = None

    def forward(self, x):
        if self.transform:
            x = self.transform(x)  # original hgclr
        n = x.shape[0]
        x = F.normalize(x, p=2, dim=1) / np.sqrt(self.tau)
        # 2B * 2B
        sim = x @ x.t()
        sim[np.arange(n), np.arange(n)] = -1e9

        logprob = F.log_softmax(sim, dim=1)

        m = 2

        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm

        return loss


class BertAndGraphModel(BertModel):
    def __init__(self, config, local_config):
        super(BertAndGraphModel, self).__init__(config)
        # print(config)
        self.config = config
        self.local_config = local_config
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
        self.text_drop = nn.Dropout(0.1)
        self.bert = BertModel(config)
        # self.bert_decoder = T5ForSequenceClassification(config)
        self.bert_pool = BertPoolingLayer('cls')
        self.structure_encoder = None

        # Parse edge list of label hierarchy
        label_hier = torch.load(os.path.join(self.local_config.data_dir, self.local_config.dataset, 'slot.pt'))
        path_dict = {}
        for s in label_hier:
            for v in label_hier[s]:
                path_dict[v] = s
        self.edge_list = [[v, i] for v, i in path_dict.items()]
        self.edge_list += [[i, v] for v, i in path_dict.items()]
        self.edge_list = torch.tensor(self.edge_list).transpose(0, 1)
        # Graph Data
        self.graph = GTData(x=None, edge_index=self.edge_list)

        self.trans_dup = nn.Sequential(nn.Linear(config.num_labels, config.num_labels),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(config.num_labels, config.num_labels),
                                       nn.Dropout(p=local_config.structure_encoder.trans_dp)
                                       )
        # For label attention
        if hasattr(local_config, "label") and local_config.label:
            self.label_type = local_config.label_type
            self.label_dict = torch.load(
                os.path.join(local_config.data_dir, local_config.dataset, 'bert_value_dict.pt'))
            self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}
            self.label_name = []
            for i in range(len(self.label_dict)):
                self.label_name.append(self.label_dict[i])
            self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
            self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)

            # self.attn = FusionLayer1(config, local_config)
            self.attn = None
            self.label_embeddings = nn.Embedding(config.num_labels, config.hidden_size)

            self.trans_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.Dropout(p=local_config.structure_encoder.trans_dp)
                                            )
        else:
            self.trans_proj = nn.Sequential(nn.Linear(config.hidden_size, local_config.hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(local_config.hidden_dim, local_config.hidden_dim),
                                            nn.Dropout(p=local_config.structure_encoder.trans_dp)
                                            )
        self.init_weights()

    def batch_duplicate(self, text_embeds, repeats=None):
        if repeats is None:
            rep = self.num_labels
        text_embeds = text_embeds.unsqueeze(1)
        text_embeds = torch.repeat_interleave(text_embeds, repeats=rep, dim=1)
        text_embeds = self.trans_proj(text_embeds)
        text_embeds = torch.transpose(text_embeds, -1, -2)
        text_embeds = self.trans_dup(text_embeds)
        text_embeds = torch.transpose(text_embeds, -1, -2)
        return text_embeds

    def align_graph(self, embeds, batch_size):
        batch_graph = [copy.deepcopy(self.graph) for _ in range(batch_size)]
        for i in range(batch_size):
            batch_graph[i].x = embeds[i]
        return batch_graph

    @staticmethod
    def extract_local_hierarchy(node_embeds, labels, node_mask):
        return torch.where(labels.unsqueeze(-1) == 0, node_mask.expand_as(node_embeds), node_embeds)

    # -----------------Freezing--------------------------
    @staticmethod
    def __children(module):
        return module if isinstance(module, (list, tuple)) else list(module.children())

    def __apply_leaf(self, module, func):
        c = self.__children(module)
        if isinstance(module, nn.Module):
            func(module)
        if len(c) > 0:
            for leaf in c:
                self.__apply_leaf(leaf, func)

    def __set_trainable(self, module, flag):
        self.__apply_leaf(module, lambda m: self.__set_trainable_attr(m, flag))

    @staticmethod
    def __set_trainable_attr(module, flag):
        module.trainable = flag
        for p in module.parameters():
            p.requires_grad = flag

    def freeze(self):
        self.__set_trainable(self.bert, False)  # freeze all params in bert

    def unfreeze_all(self):
        self.__set_trainable(self.bert, True)

    def unfreeze(self, start_layer, end_layer, pooler=True):
        """
        # Unfreeze the params in bert.encoder ranged from $[start_layer, end_layer]$
        while keeping other parameters in bert freeze.
        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        """
        self.__set_trainable(self.bert, False)
        for i in range(start_layer, end_layer + 1):
            self.__set_trainable(self.bert.encoder.layer[i], True)
        if self.bert.pooler is not None:
            self.__set_trainable(self.bert.pooler, pooler)


class BertAndCodingTreeModel(BertAndGraphModel):
    def __init__(self, config, local_config):
        super(BertAndCodingTreeModel, self).__init__(config, local_config)
        # Coding tree
        self.tree, self.fb_keys = self.construct_coding_tree()
        self.init_weights()

    def construct_coding_tree(self):
        tree = GTData(x=None, edge_index=self.edge_list)
        adj = to_dense_adj(self.edge_list, max_num_nodes=self.num_labels).squeeze(0)
        nodeSize, edgeSize, edgeMat = get_tree_data(adj, self.local_config.tree_depth)
        tree.treeNodeSize = torch.LongTensor(nodeSize).view(1, -1)
        for layer in range(1, self.local_config.tree_depth + 1):
            tree['treePHLayer%s' % layer] = torch.ones([nodeSize[layer], 1])  # place holder
            tree['treeEdgeMatLayer%s' % layer] = torch.LongTensor(edgeMat[layer]).T
        fb_keys = [key for key in tree.keys() if key.find('treePHLayer') >= 0]
        return tree, fb_keys

    def align_tree(self, embeds, batch_size):
        batch_tree = [copy.deepcopy(self.tree) for _ in range(batch_size)]
        # debug(self.tree)
        for i in range(batch_size):
            batch_tree[i].x = embeds[i]
        return batch_tree


class StructureContrast(BertAndCodingTreeModel):
    def __init__(self, config, local_config):
        """
        HILL: Hierarchy-aware Information Lossless contrastive Learning
        """
        super(StructureContrast, self).__init__(config, local_config)
        self.contrastive_lossfct = NTXent(config, tau=local_config.contrast.tau, transform=False)
        self.cls_loss = local_config.cls_loss  # Whether to use classification loss
        self.contrast_loss = local_config.contrast_loss  # Whether to use contrastive loss
        self.multi_label = local_config.multi_label
        self.lamda = local_config.lamda  # weight of contrastive loss

        self.structure_encoder = HRLEncoder(local_config)
        self.output_type = local_config.hrl_output
        hidden_size = config.hidden_size if not self.output_type == 'concat' else config.hidden_size * 2
        self.classifier = nn.Linear(hidden_size, config.num_labels)  # structure_encoder.output_dim := hidden_size

        self.text_proj = nn.Sequential(nn.Linear(config.hidden_size, local_config.contrast.proj_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(local_config.contrast.proj_dim, local_config.contrast.proj_dim)
                                       )
        self.tree_proj = nn.Sequential(
            nn.Linear(local_config.structure_encoder.output_dim, local_config.contrast.proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(local_config.contrast.proj_dim, local_config.contrast.proj_dim)
        )
        self.node_mask = nn.Parameter(torch.randn(local_config.hidden_dim))
        self.init_weights()  # Warning! This is NOT training BERT from scratch

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_cls_embed = self.text_drop(self.bert_pool(hidden_states))
        # decoder = self.bert_decoder()
        batch_size = hidden_states.shape[0]
        loss = 0
        contrast_logits = None
        text_embeds = self.batch_duplicate(pooled_cls_embed)
        # debug(text_embeds.shape)
        batch_tree = self.align_tree(text_embeds, batch_size)
        # debug(len(batch_tree))
        tree_loader = DataLoader(batch_tree, batch_size=batch_size, follow_batch=self.fb_keys)
        # debug(tree_loader)
        contrast_output = self.structure_encoder(next(iter(tree_loader)))
        self.output_type = "residual"
        self.contrast_loss = True
        if self.output_type == 'tree':
            logits = self.classifier(contrast_output)  # hill
        elif self.output_type == 'residual':
            logits = self.classifier(pooled_cls_embed + contrast_output)  # hill + bert
        elif self.output_type == 'concat':
            logits = self.classifier(torch.cat([pooled_cls_embed, contrast_output], dim=1))  # [bert, hill]
        else:
            logits = self.classifier(pooled_cls_embed)  # bert
        self.multi_label = True
        if labels is not None:
            if self.training:
                if not self.multi_label:
                    loss_fct = CrossEntropyLoss()
                    target = labels.view(-1)
                    loss += loss_fct(logits.view(-1), target)
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    target = labels.to(torch.float32)
                    # debug(target)
                    loss += loss_fct(logits.view(-1, self.num_labels), target)
                if self.contrast_loss:
                    contrastive_loss = self.contrastive_lossfct(
                        torch.cat([self.text_proj(pooled_cls_embed), self.tree_proj(contrast_output)], dim=0), )
                    loss += contrastive_loss * self.lamda

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'contrast_logits': contrast_logits,
        }
