# 作者: 宇亮
# 2026年03月16日21时02分08秒
# Julian_guo153@163.com

from transformers import BertModel, BertTokenizer

bert = BertModel.from_pretrained("bert-base-chinese")

# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

my_input = "我爱你"
my_output = tokenizer(my_input, truncation=True, padding="max_length", max_length=128)
print(my_output)

if __name__ == '__main__':
    # print(get_parameter_number(bert))
    # emb_num = 21128*768 + 2*768 + 512*768
    # self_att_num = 768*768*3 + 768*768 + 768*3072 + 3072*768
    # pooler_num = 768*768
    # print(emb_num + self_att_num * 12 + pooler_num)
    # for name, param in bert.named_parameters():
    #     print(name, param.shape)
    pass
