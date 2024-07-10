import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

len_data = 0


def preprocess_vn_express():
    df = pd.read_csv('data-vn-express.clean.csv', sep='\t', encoding='utf-8')
    df = df.dropna()
    data = []
    global len_data
    len_data = len(df.sentences.to_list())
    for index, row in df.iterrows():
        sample_text = row['sentences']
        sample_label = [row.lv1, row.lv2]
        data.append({
            'doc_token': sample_text,
            'doc_label': sample_label,
            'doc_topic': [],
            'doc_keyword': []})
    f = open('wos_total.json', 'w', encoding='utf-8')
    for line in data:
        line = json.dumps(line, ensure_ascii=False)
        f.write(line + '\n')
    f.close()


def split_train_dev_test():
    global len_data
    f = open('wos_total.json', 'r')
    data = f.readlines()
    f.close()
    id = [i for i in range(len_data)]
    np_data = np.array(data)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = list(train)
    val = list(val)
    test = list(test)
    f = open('wos_train.json', 'w')
    f.writelines(train)
    f.close()
    f = open('wos_test.json', 'w')
    f.writelines(test)
    f.close()
    f = open('wos_val.json', 'w')
    f.writelines(val)
    f.close()

    print(len(train), len(val), len(test))
    return


def get_hierarchy():
    f = open('wos_total.json', 'r')
    data = f.readlines()
    f.close()
    label_hierarchy = {}
    label_hierarchy['Root'] = []
    for line in data:
        line = line.rstrip('\n')
        line = json.loads(line)
        line = line['doc_label']
        if line[0] in label_hierarchy:
            if line[1] not in label_hierarchy[line[0]]:
                label_hierarchy[line[0]].append(line[1])
        else:
            label_hierarchy['Root'].append(line[0])
            label_hierarchy[line[0]] = [line[1]]
    f = open('wos.taxnomy', 'w')
    for i in label_hierarchy.keys():
        line = [i]
        line.extend(label_hierarchy[i])
        line = '\t'.join(line) + '\n'
        f.write(line)
    f.close()


if __name__ == '__main__':
    preprocess_vn_express()
    get_hierarchy()
    split_train_dev_test()