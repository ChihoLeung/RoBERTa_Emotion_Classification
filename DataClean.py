import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba
plt.style.use('seaborn')
sns.set(font_scale=2)

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号

CSV_PATH = 'D:/.../roberta-masterContent/data/' #数据集所在具体路径

# 转换编码
#   将编码为ANSI的csv文件转换为UTF-8，方便后续进行pd操作
def re_encode(path):    # 定义编码转换函数
    with open(path, 'r', encoding='GB2312', errors='ignore') as file:
        lines = file.readlines()
    with open(path, 'w', encoding='utf-8') as file:
        file.write(''.join(lines))
        
re_encode(CSV_PATH+'raw/nCoV_100k_train.labled.csv') # 数据集文件路径
# re_encode(CSV_PATH+'raw/nCov_10k_test.csv') # 数据集文件路径


# 数据预处理
#   对句子进行中文分词
def stopwordslist():
    # 创建一个停用词列表
    stopwords = [line.strip() for line in open('hit_stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    # stopwords = [line.strip() for line in open('hit_stopwords.txt',encoding='UTF-8').readlines()]
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ""
    return outstr

#   去除转发用户名——“//@username:”或“回复@username:”
def remove_username(src):
    vTEXT = re.sub(r'\/\/@(\w|\-|[\u4e00-\u9fa5])*(:|：)?', '', src, flags=re.MULTILINE)
    vTEXT1 = re.sub(r'回复@(\w|\-|[\u4e00-\u9fa5])*(:|：)?', '', vTEXT, flags=re.MULTILINE)
    return vTEXT1

#   去除URL
def remove_url(src):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', src, flags=re.MULTILINE)
    return vTEXT

#   去除无意义关键字
def remove_illegal(src):
    comment = []
    for com_str in src:
        com_str = remove_username(com_str)  # 去除转发用户名
        com_str = remove_url(com_str)   # 去除URL
        com_str = com_str.replace(',','，')
        com_str = com_str.replace("展开全文c",'')
        com_str = com_str.replace("@QQ音乐?",'')
        com_str = com_str.replace("（@网易云音乐）",'')
        com_str = com_str.replace("(@网易云音乐)",'')
        com_str = com_str.replace("O网页链接",'')
        com_str = seg_depart(com_str)
        if not com_str:
            com_str = '无'
        comment.append(com_str)
    vTEXT = comment
    return vTEXT

#   训练集数据清洗
def clean(path):
    # 读取数据
    print("---读取数据中...---")
    df_data = pd.read_csv(path, usecols=['微博中文内容', '情感倾向'], encoding='utf-8', engine ='python')
    # df_data = pd.read_csv(path, usecols=['微博发布时间', '微博中文内容', '情感倾向'], encoding='utf-8', engine ='python')
    # df_data['微博发布时间'] = pd.to_datetime('2020年' + df_data['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')
    print("---读取数据成功---")
    
    print("数据矩阵形状：",df_data.shape)
    print("数据矩阵列标签：",df_data.columns)
    print("数据矩阵前5行：")
    print(df_data.head())
    
    # 文本长度差异
    # 查看训练集各个变量的类型，数量等信息，同时查看标签数据
    print("---文本长度差异---")
    print("数据矩阵每列的内存使用情况：")
    print(df_data.info(memory_usage='deep'))
    print("数据矩阵中“情感倾向”列的所有值及数量：")
    print(df_data['情感倾向'].value_counts(dropna=False))
    
    # 数据清洗
    print("---数据清洗中...---")
    df_data = df_data[df_data['微博中文内容'].notna()]   # 将“微博中文内容”中空白值去除
    df_data['微博中文内容'] = remove_illegal(df_data['微博中文内容']) # 去除无意义关键字
    df_data = df_data[df_data['情感倾向'].isin(['-1','0','1'])]  # 将label异常值去除
    df_data['情感倾向'] = df_data['情感倾向'].astype(np.int32)    # 将label转化为整型
    
    print("数据矩阵清洗后每列的内存使用情况：")
    print(df_data.info(memory_usage='deep'))
    print("数据矩阵清洗后“情感倾向”列的所有值及数量：")
    print(df_data['情感倾向'].value_counts(dropna=False))
    print("数据矩阵前5行（清洗后）：")
    print(df_data.head())
    
    # 保存清理后的数据
    df_data.to_csv(CSV_PATH+"data_clean.csv", encoding='utf-8', index=None)
    print("---数据清洗完成---")

clean(CSV_PATH+'raw/nCoV_100k_train.labled.csv')

#   测试集数据清洗
def test_dataclean(path):
    # 读取数据
    print("---读取数据中...---")
    # df_data = pd.read_csv(path, usecols=['微博中文内容', '情感倾向'], encoding='utf-8', engine ='python')
    df_data = pd.read_csv(path, usecols=['微博发布时间', '微博中文内容', '情感倾向'], encoding='utf-8', engine ='python')
    # df_data['微博发布时间'] = pd.to_datetime('2020年' + df_data['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')
    print("---读取数据成功---")
    
    print("数据矩阵形状：",df_data.shape)
    print("数据矩阵列标签：",df_data.columns)
    print("数据矩阵前5行：")
    print(df_data.head())
    
    # 文本长度差异
    # 查看训练集各个变量的类型，数量等信息，同时查看标签数据
    print("---文本长度差异---")
    print("数据矩阵每列的内存使用情况：")
    print(df_data.info(memory_usage='deep'))
    
    # 数据清洗
    print("---数据清洗中...---")
    df_data = df_data[df_data['微博中文内容'].notna()]   # 将“微博中文内容”中空白值去除
    df_data['微博中文内容'] = remove_illegal(df_data['微博中文内容']) # 去除无意义关键字
    
    print("数据矩阵清洗后每列的内存使用情况：")
    print(df_data.info(memory_usage='deep'))
    print("数据矩阵前5行（清洗后）：")
    print(df_data.head())
    
    # 保存清理后的数据
    df_data.to_csv(CSV_PATH+"results.csv", encoding='utf-8', index=None)
    print("---数据清洗完成---")

# test_dataclean(CSV_PATH+'raw/nCov_10k_test.csv')
