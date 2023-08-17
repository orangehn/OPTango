import torch.nn as nn


def init_weight(net):
    # from transformers.models.bert.modeling_bert import BertLayer, BertEncoder, BertEmbeddings
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):  # NonDynamicallyQuantizableLinear?
            nn.init.normal_(m.weight, 0, 0.01)
            # nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.01)
            # nn.init.xavier_uniform_(m.weight)
        else:
            pass
