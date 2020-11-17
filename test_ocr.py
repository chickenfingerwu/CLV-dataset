import easyocr

reader = easyocr.Reader(['ja'])
result = reader.readtext('./Datasets/ETL_test/0x59be/599448.png')
print(result)