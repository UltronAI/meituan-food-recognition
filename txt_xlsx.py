
import xlwt


# cell_overwrite_ok=True
# sheet.write(“infoPlist”,cell_overwrite_ok=True)

f=open('test_prediction.txt','r')
str=f.readlines()

book=xlwt.Workbook()
sheet=book.add_sheet('prediction')

row=1


sheet.write(0,0,'id')
sheet.write(0,1,'predicted')


for line in str:
    image_id=line.split(' ',1)[0]
    label=line.split(' ',1)[1]
    label1=label.split('[')[1]
    label2=label1.split(']')[0]

    predict1 = label2.split(', ')[0]
    predict2 = label2.split(', ')[1]
    predict3 = label2.split(', ')[2]
    predict=predict1+' '+predict2+' '+predict3
    sheet.write(row,0,image_id)
    sheet.write(row,1,predict)
    # sheet.write(row,1,predict2)
    # sheet.write(row,1,predict3)
    row+=1

book.save('predcition12.xls')

