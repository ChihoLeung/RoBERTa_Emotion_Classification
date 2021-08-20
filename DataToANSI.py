CSV_PATH = 'D:/.../roberta-masterContent/data/' #数据集所在具体路径

# 转换编码
# 将编码UTF-8的csv文件转换为ANSI
def re_encode(path):    # 定义编码转换函数
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
    with open(path, 'w', encoding='GB2312') as file:
        file.write(''.join(lines))
        
re_encode(CSV_PATH+'results.csv') # 写自己文件的路径