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
# 将编码为ANSI的csv文件转换为UTF-8，方便后续进行pd操作
def re_encode(path):    # 定义编码转换函数
    with open(path, 'r', encoding='GB2312', errors='ignore') as file:
        lines = file.readlines()
    with open(path, 'w', encoding='utf-8') as file:
        file.write(''.join(lines))
        
# re_encode(CSV_PATH+'results.csv') # 数据集文件路径

# 可视化
print("---开始进行数据可视化---")
df_data = pd.read_csv(CSV_PATH+"results.csv", encoding='utf-8', engine ='python')
print("前10行数据：")
print(df_data.head(10))

df_data['content_len'] = df_data['微博中文内容'].str.len()
info = df_data['content_len'].describe(include='all')
print("“微博中文内容”列字数统计：")
print(info)

# 情感倾向标签总体分布可视化
print("“情感倾向”列的所有值及数量：")
print(df_data['情感倾向'].value_counts(dropna=False))
print("“情感倾向”列的各个值的数量占比：")
print(df_data['情感倾向'].value_counts(dropna=False)/df_data['情感倾向'].count())
# (df_data['情感倾向'].value_counts(dropna=False)/df_data['情感倾向'].count()).plot.bar()
(df_data['情感倾向'].value_counts(dropna=False)).plot.bar()
plt.xlabel('情感倾向标签',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('标签总体分布可视化',fontsize=15)
plt.tight_layout()
plt.show()

# 感情倾向的数量变化和占比变化情况
df_data['time'] = pd.to_datetime('2020年' + df_data['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')
df_data['date'] = df_data['time'].dt.date # 转换日期格式

# 对数据按照日期和情感倾向进行分类
date_influence = df_data.groupby(['date','情感倾向'],as_index=False).count() 

# sns.relplot(x="date", y="微博中文内容",kind='line', hue='情感倾向',palette=["b", "r",'g'],data=date_influence)
sns.relplot(x="date", y="微博中文内容",kind='line', hue='情感倾向', style='情感倾向', palette=["b", "r",'g'],data=date_influence)
plt.xticks(rotation=45,fontsize=12)
plt.xlabel('日期',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('微博数量分布图',fontsize=15)
plt.show()

date_influence = date_influence.merge(df_data.groupby('date',as_index=False)['情感倾向'].count().rename(columns={'情感倾向':'weibo_count'}),how='left',on='date')
date_influence['weibo_rate'] = date_influence['微博中文内容']/date_influence['weibo_count']

# sns.relplot(x="date", y="weibo_rate", kind="line", hue='情感倾向',palette=["b", "r",'g'],data=date_influence)
sns.relplot(x="date", y="weibo_rate", kind="line", hue='情感倾向', style='情感倾向', palette=["b", "r",'g'],data=date_influence)
plt.xticks(rotation=45,fontsize=12)
plt.xlabel('日期',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('微博情感占比分布图',fontsize=15)
plt.show()

# 评论的长度分布
df_data['char_length'] = df_data['微博中文内容'].astype(str).apply(len) #计算每条微博评论的长度
sns.distplot(df_data['char_length'],kde=False)
plt.xlabel('长度',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('评论长度分布',fontsize=15)
plt.tight_layout()
plt.show()

""" def label(data):
    if data <125:
        return '小于125'
    elif data<150 and data>125:
        return '125-150'
    else:
        return '大于150'
df_data['length_label']=df_data['char_length'].apply(label)
sns.countplot('情感倾向',hue='length_label',data=df_data)
plt.xlabel('长度',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('清洗前评论长度分布与情感倾向的关系',fontsize=15)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show() """