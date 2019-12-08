import numpy as np

models = [
    'wide_resnet50_2',
    'resnet50',
    'resnext50_32x4d',
    'se_resnext101_32x4d'
]

data = {}

for model_name in models:
    data[model_name] = {}
    with open('{}.txt'.format(model_name), 'r') as f:
        last_class = -1
        class_acc = 0
        for line in f.readlines():
            image_name = line.strip().split()[0]
            class_id = int(image_name.split('_')[0])

            if class_id != last_class:
                data[model_name][class_id] = {
                    'acc': 0.0,
                    'error': {},
                    'pred': {}
                }
                last_class = class_id

            pred1 = int(line.strip().split()[1].strip(' ,[]'))
            pred2 = int(line.strip().split()[2].strip(' ,[]'))
            pred3 = int(line.strip().split()[3].strip(' ,[]'))
            pred_top3 = [pred1, pred2, pred3]
            if class_id in pred_top3:
                data[model_name][class_id]['acc'] += 0.1
                if pred1 in data[model_name][class_id]['pred'].keys():
                    data[model_name][class_id]['pred'][pred1] += 1
                else:
                    data[model_name][class_id]['pred'][pred1] = 1
            else:
                if pred1 in data[model_name][class_id]['error'].keys():
                    data[model_name][class_id]['error'][pred1] += 1
                else:
                    data[model_name][class_id]['error'][pred1] = 1

    sorted_data = sorted(data[model_name], key=lambda k: data[model_name][k]['acc'], reverse=True)
    print(model_name)
    for i in range(10):
        cls_id = sorted_data[i]
        acc = data[model_name][cls_id]['acc']
        # print('[{:3d}] acc:{:.1f} error:{}'.format(cls_id, acc, data[model_name][cls_id]['error']))
        print('[{:3d}] acc:{:.1f} pred:{}'.format(cls_id, acc, data[model_name][cls_id]['pred']))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')