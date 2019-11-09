import os
import os.path

train_dir=('C:/Users/Administrator/Desktop/MTFood-1000/train')
train_txt=('C:/Users/Administrator/Desktop/MTFood-1000/train_label.txt')
val_dir=('C:/Users/Administrator/Desktop/MTFood-1000/val')
val_txt=('C:/Users/Administrator/Desktop/MTFood-1000/val_label.txt')
test_dir=('C:/Users/Administrator/Desktop/MTFood-1000/test')
test_txt=('C:/Users/Administrator/Desktop/MTFood-1000/test_label.txt')


for i,j,k in os.walk(train_dir):
    # print(i,j,k)
    with open(train_txt,'a') as f:
        for h in k:
            label=(h.split("_"))[0]
            s1=h +' '+ label +'\n'
            f.write(s1)
    f.close()


for i,j,k in os.walk(val_dir):
    # print(i,j,k)
    with open(val_txt,'a') as f:
        for h in k:
            label=(h.split("_"))[0]
            s1=h+' '+label +'\n'
            f.write(s1)
    f.close()

for i,j,k in os.walk(test_dir):
    # print(i,j,k)
    with open(test_txt,'a') as f:
        for h in k:
            label=str(0)
            s1=h +' '+ label +'\n'
            f.write(s1)
    f.close()
