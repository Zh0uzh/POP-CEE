from simpletransformers.classification.classification_utils import (
    InputExample,
    LazyClassificationDataset,
    convert_examples_to_features,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def process_prompt(logits, input_ids, prompt_labels_ids, tokenizer, emotion, position):
    mask_position = np.argwhere(input_ids.cpu()==tokenizer.mask_token_id)
    # print('self.tokenizer.mask_token_id', self.tokenizer.mask_token_id)
    # print('mask_position', mask_position)
    # print('mask_position.shape', mask_position.shape)
    # print('emotion', emotion)
    # print('position', position)
    
    binary_logits = []
    mask_index = 0
    for i in range(input_ids.shape[0]):
        # print('position[i][0] - position[i][1] == 0 or (emotion == 1 and position[i][0] - position[i][1] == 1)', position[i][0] - position[i][1] == 0 or (emotion[i] == 1 and position[i][0] - position[i][1] == 1))
        if position[i][0] - position[i][1] == 0:    # 句子本身
            sum_binary_logits = []
            for j in range(0, 5, 4):
                # if mask_index >= mask_position.shape[1]:
                #     # mask_index = mask_position.shape[1] - 1
                #     print('mask_position', mask_position)
                #     print('mask_position.shape', mask_position.shape)
                #     print('mask_index', mask_index)
                sum_binary_logits.append(logits[mask_position[0, mask_index], mask_position[1, mask_index], mask_position[2, mask_index], prompt_labels_ids[j]])
                mask_index += 1
            sum_binary_logits = torch.stack(sum_binary_logits, dim=0)
            sum_binary_logits = torch.sum(sum_binary_logits, dim=0)
            # print('sum_binary_logits', sum_binary_logits)
            binary_logits.append(sum_binary_logits / 2)
        elif emotion[i] == 1 and position[i][0] - position[i][1] == 1:       # surprise
            sum_binary_logits = []
            for j in range(1, 5, 3):
                # if mask_index >= mask_position.shape[1]:
                #     # mask_index = mask_position.shape[1] - 1
                #     print('mask_position', mask_position)
                #     print('mask_position.shape', mask_position.shape)
                #     print('mask_index', mask_index)
                sum_binary_logits.append(logits[mask_position[0, mask_index], mask_position[1, mask_index], mask_position[2, mask_index], prompt_labels_ids[j]])
                mask_index += 1
            sum_binary_logits = torch.stack(sum_binary_logits, dim=0)
            sum_binary_logits = torch.sum(sum_binary_logits, dim=0)
            # print('sum_binary_logits', sum_binary_logits)
            binary_logits.append(sum_binary_logits / 2)
        elif emotion[i] == 0:       # happiness
            sum_binary_logits = []
            for j in range(2, 5, 2):
                # if mask_index >= mask_position.shape[1]:
                #     # mask_index = mask_position.shape[1] - 1
                #     print('mask_position', mask_position)
                #     print('mask_position.shape', mask_position.shape)
                #     print('mask_index', mask_index)
                sum_binary_logits.append(logits[mask_position[0, mask_index], mask_position[1, mask_index], mask_position[2, mask_index], prompt_labels_ids[j]])
                mask_index += 1
            sum_binary_logits = torch.stack(sum_binary_logits, dim=0)
            sum_binary_logits = torch.sum(sum_binary_logits, dim=0)
            # print('sum_binary_logits', sum_binary_logits)
            binary_logits.append(sum_binary_logits / 2)
        else:
            sum_binary_logits = []
            for j in range(3, 5, 1):
                # if mask_index >= mask_position.shape[1]:
                #     # mask_index = mask_position.shape[1] - 1
                #     print('mask_position', mask_position)
                #     print('mask_position.shape', mask_position.shape)
                #     print('mask_index', mask_index)
                sum_binary_logits.append(logits[mask_position[0, mask_index], mask_position[1, mask_index], mask_position[2, mask_index], prompt_labels_ids[j]])
                mask_index += 1
            sum_binary_logits = torch.stack(sum_binary_logits, dim=0)
            sum_binary_logits = torch.sum(sum_binary_logits, dim=0)
            # print('sum_binary_logits', sum_binary_logits)
            binary_logits.append(sum_binary_logits / 2)
    binary_logits = torch.stack(binary_logits, dim=0)

    assert mask_index == mask_position.shape[1]
    # print('binary_logits', binary_logits)
    # print('binary_logits.shape', binary_logits.shape)

    return binary_logits

def prompt_outputs(logits, input_ids, prompt_labels_ids, tokenizer, labels, emotion, position):

    # print('logits', logits.shape)
    # print('input_ids.shape', input_ids.shape)

    binary_logits = process_prompt(logits, input_ids, prompt_labels_ids, tokenizer, emotion, position)

    binary_logits = position_process(binary_logits, emotion, position)

    binary_logits = F.softmax(binary_logits, dim=-1)


    # 计算损失
    loss_fn = nn.CrossEntropyLoss()

    # print('binary_logits', binary_logits.shape)
    # print('labels', labels)
    loss = loss_fn(binary_logits, labels)
    
    return  loss, binary_logits

def project_examples_with_prompt(examples, tokenizer):
    # templete = " is <mask> related to "
    # prompt_labels = [
    #     ["not", ""],
    #     ["no", "last"],
    #     ["No", "Yes"]
    # ]
    new_examples = examples
    # prompt_label_ids = []
    # prompt_label_ids = [
    #     [3084, 9904],
    #     [3084, 9904],
    #     [3084, 9904],
    #     [751, 1025],    # [outside, inside]
    #     [172, 98],      # ["then", "so"]
    # ]
    prompt_label_ids = [
        [45, 1437],     # ["not", ""]
        [117, 94],      # ["no", "last"]
        [172, 98],      # ["then", "so"]
        [8, 4634],        # ["and", "thus"]
        [3084, 9904]    # ["No", "Yes"]
    ]

    # # 将example添加prompt
    # for each_example in examples:
    #     pos = [i for i in range(len(each_example.text_a)) if each_example.text_a.startswith('<SEP>', i)]
    #     # print("tokenizer.sep_token", tokenizer.sep_token)
    #     # print("pos", pos)
    #     new_text_a = each_example.text_a[:pos[0]] + '<SEP> ' + '"' + \
    #         each_example.text_a[(pos[0]+5):pos[1]] + '"' + templete + '"' + each_example.text_a[(pos[1]+5):pos[2]] + '"' + \
    #             " according to dialog " + '"' + each_example.text_a[(pos[2]+5):] + ' "'
    #     new_examples.append(InputExample(each_example.guid, new_text_a, None, each_example.label))



    # for labels in prompt_labels:
    #     prompt_label_id = []
    #     for label in labels:
    #         text_replaced = new_examples[0].text_a
    #         mask_pos = tokenizer.encode(text_replaced, add_special_tokens=False).index(tokenizer.mask_token_id)
    #         # print('mask_pos', mask_pos)     
    #         text_replaced = text_replaced.replace(tokenizer.mask_token, label)
    #         # print('tokenizer.mask_token_id', tokenizer.mask_token_id)
    #         # print('text_replaced', text_replaced)
    #         text_replaced_ids = tokenizer.encode(text_replaced, add_special_tokens=False)
    #         # print('text_replaced_ids', text_replaced_ids)
    #         prompt_label_id.append(text_replaced_ids[mask_pos])
    #     prompt_label_ids.append(prompt_label_id)
    # print('prompt_label_ids', prompt_label_ids)

    # for each_example in new_examples:
    #     print('guid', each_example.guid)
    #     print('text_a', each_example.text_a)
    # #     print('label', each_example.label)
    # print('new_examples[4].text_a', new_examples[2].text_a[0])
    # print('new_examples[4].text_a', new_examples[2].text_a[1])
    # print('new_examples[4].label', new_examples[2].label)
    # print('new_examples[4].text_a', type(new_examples[2].text_a))

    return new_examples, prompt_label_ids

def position_process(logits, emotion, position):
    for j in range(emotion.shape[0]):
        if position[j][0]-position[j][1] > 8:
            logits[j][0] += 1e3
            logits[j][1] -= 1e3
        elif emotion[j] == 1 and position[j][0]-position[j][1] > 1:
            logits[j][0] += 1e3
            logits[j][1] -= 1e3
        # elif emotion[j] == 1 and position[j][0]-position[j][1] == 1:
        #     logits[j][0] -= 1e3
        elif emotion[j] == 4 and emotion[j] == 5 and (position[j][0]-position[j][1]) % 2 == 0:
            logits[j][0] += 1e3
            logits[j][1] -= 1e3
    return logits