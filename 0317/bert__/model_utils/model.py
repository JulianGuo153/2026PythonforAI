# 作者: 宇亮
# 2026年03月17日15时37分12秒
# Julian_guo153@163.com

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

class MyBERTModel(nn.Module):
    def __init__(self, bert_path, num_classes, device):
        super(MyBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)

        # config = BertConfig.from_pretrained(bert_path)
        # self.bert = BertModel(config)


        self.device = device
        self.cls_head = nn.Linear(768, num_classes)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def forward(self, text):
        my_input = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        my_input_ids = my_input["input_ids"].to(self.device)
        my_attention_mask = my_input["attention_mask"].to(self.device)
        my_token_type_ids = my_input["token_type_ids"].to(self.device)
        sequence_out, pooler_out = self.bert(input_ids=my_input_ids,
                           attention_mask=my_attention_mask,
                           token_type_ids=my_token_type_ids,
                           return_dict=False)
        pred = self.cls_head(pooler_out)
        return pred


if __name__ == '__main__':
    my_model = MyBERTModel("../bert-base-chinese", 2)
    pred = my_model("今天天气真好")
    pass
