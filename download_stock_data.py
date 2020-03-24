import pymysql
import tushare as ts
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

token = 'e8df84bd1b25a8a2a2ceb7edf7ad41f2c3a1d3ec604bb8abd40321f4'
ts.set_token(token)
pro = ts.pro_api()
# 获取上证50（2019年12月31号当天的成分股共33个具有参考价值）\
def get_sz50():
    data = pro.stock_basic(exchange='SSE', list_status='L')
    df = pro.index_weight(index_code='000016.SH', trade_date='20191231')
    ld = []
    for ts in df.con_code:
        ld.append(data[data.ts_code == ts].list_date.values[0])
    df['list_date'] = ld
    df = df[df.list_date < '20080101']
    df.index = range(len(df))
    return df.con_code


# 获取数据
def get_data(ts_code):
    ts_code = ts_code
    start_date = '20071217'
    end_date = '20191231'
    daily_basic = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date \
                                  , fields='turnover_rate,volume_ratio,pe,pb,ps,ps_ttm')
    pro_bar = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date, ma=[5, 10])
    df = pd.concat([pro_bar.iloc[:-10, 1:], daily_basic.iloc[:-10, 1:]], axis=1, sort=True)
    if df.isna().any().any():
        df = df.fillna(0)
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: int(x * 100) / 100)
    column_name = df.columns
    column_create = ''
    column_insert = '('
    for co in column_name:
        if co == 'trade_date':
            column_create += co.upper() + '_' + ' VARCHAR(100),\n'
        else:
            column_create += co.upper() + '_' + ' DOUBLE,\n'
        column_insert += co.upper() + '_,'
    column_create += 'PRIMARY KEY (TRADE_DATE_)'
    column_insert += ')'
    trade_cal = pro.trade_cal(exchange='SSE', start_date='20080101', end_date=end_date, is_open='1').cal_date
    return column_create, column_insert, df, trade_cal


# 保存至数据库
def save_to_datebase(ts_code, column_create, column_insert, df, trade_cal):
    print('Start saving {}...'.format(ts_code))
    db = pymysql.connect("localhost", "root", "123456", "stock")
    cursor = db.cursor()
    try:
        cursor.execute('CREATE TABLE {} ({})'.format('SH_' + ts_code[:-3], column_create))
        print('CREATE TABLE {} SUCCESS'.format('SH_' + ts_code[:-3]))
    except:
        print('FAIL TO CREATE TALBE {}'.format('SH_' + ts_code[:-3]))
    for ix, cal in enumerate(trade_cal):
        #         print(df)
        data = df[df.trade_date == cal]
        #         print(ix)
        #         print(trade_cal[ix])
        #         print(data)
        if len(data) == 0:
            now = ix - 1
            while len(data) == 0:
                data = df[df.trade_date == trade_cal.values[now]]
                now -= 1
            data.trade_date = cal
            data = tuple(data.values[0])
        else:
            data = tuple(data.values[0])

        #         print('data',data)
        try:
            cursor.execute('INSERT INTO {} {}) VALUES {}'.format('SH_' + ts_code[:-3], column_insert[:-2], data))
            #             print('INSERT INTO {} {}) VALUES {}'.format('SH_'+ts_code[:-3],column_insert[:-2],data))
            db.commit()
        except:
            # 发生错误时回滚
            print('fail to commit')
            db.rollback()
    #     data = cursor.fetchone()
    db.close()
    print('Saving {} complete...'.format(ts_code))

sz50 = get_sz50()
for ts_code in sz50:
    column_create, column_insert, df, trade_cal = get_data(ts_code=ts_code)
    save_to_datebase(ts_code,column_create,column_insert,df,trade_cal)