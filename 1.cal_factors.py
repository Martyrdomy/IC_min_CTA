# %%
import pandas as pd
import numpy as np
import pyTSL as pt
import os
import datetime as dt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from Config import factor_config as fc

head_future = pd.read_excel(os.path.join(fc.head_path, 'IC_head_future.xlsx')) # 每日主流合约

def read_file(date):
    '''
    根据日期读取对应主头合约文件
    '''
    read_date = date.strftime('%Y%m%d')
    next_idx = head_future[head_future['trade_date'] == date].index
    if next_idx == 0:
        return None
    read_future = head_future.iloc[next_idx - 1]['IC'].values[0]
    read_file = read_future + '_' + read_date + '.tdf'
    return read_file


# %%
def cal_alpha001(start_date, end_date, start_minute, corr_n, path):
    '''
    成交量差分排名与收盘涨跌幅排名的相关系数\n
    nan行数=corr_n - 1
    '''
    factor_name = 'alpha001_start{a}_corr{b}'.format(a=start_minute, b=corr_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):

        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]
        data = data[data['minute'].dt.time < dt.time(15, 00)]

        # 按分钟分块
        # min_list = pd.date_range(dt.time(9, 30), )
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算因子
        for i in range(len(min_list)):

            min_ = min_list[i]
            res = pd.DataFrame()

            if i >= corr_n:

                period_data = pd.concat(mindata_list[i - corr_n + 1: i + 1])

                # 每秒数据合并
                period_vwap = period_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
                period_vwap = pd.DataFrame(period_vwap, columns=['vwap'])
                period_vwap['vol'] = period_data.groupby('date')['vol'].sum()

                # 计算因子
                period_vwap['return'] = period_vwap['vwap'].pct_change()
                period_vwap['rank_return'] = period_vwap['return'].rank(pct=True)
                period_vwap['rank_diff_vol'] = np.log(period_vwap['vol']).diff().rank(pct=True)
                res.loc[min_, factor_name] = period_vwap[['rank_return', 'rank_diff_vol']].corr().iloc[0, 1]
            
            else:

                res.loc[min_, factor_name] = np.nan

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)
        result_list.append(res_df)

    # 保存计算时间段结果
    result_df = pd.concat(result_list)
    result_df = result_df.reset_index(names=['minute'])

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    # 结果输出
    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha002(start_date, end_date, start_minute, path):
    """
    价格跌幅与振幅之差和高低价差的比率的差分\n
    nan行数=1
    """
    factor_name = 'alpha002_start{a}'.format(a=start_minute)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = []

        # 计算因子 
        for i in range(len(min_list)):

            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]

            # 保存每分钟结果
            res_list.append(res)

        res_df = pd.concat(res_list)

        # 计算因子
        res_df[factor_name] = (((res_df['close'] - res_df['low']) - (res_df['high'] - res_df['close'])) / (res_df['high'] - res_df['low'])).diff()
        res_df[factor_name] = res_df[factor_name].fillna(0)
        res_df = res_df.reset_index(names=['minute'])

        # 保存每天结果
        result_list.append(res_df)

    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))
  

def cal_alpha003(start_date, end_date, start_minute, roll_n, path):
    """
    前10min天收盘价与条件价格之差的和\n
    nan行数=roll_n - 1
    """
    factor_name = 'alpha003_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算因子
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['pre_close'] = res_df['close'].shift()
        condition2 = res_df['close'] > res_df['pre_close']
        condition3 = res_df['close'] < res_df['pre_close']
        part2 = (res_df['close'] - np.minimum(res_df['pre_close'][condition2], res_df['low'][condition2]))
        part3 = (res_df['close'] - np.maximum(res_df['pre_close'][condition3], res_df['low'][condition3]))
        part2 = part2.fillna(0)
        part3 = part3.fillna(0)

        res_df[factor_name] = (part2 + part3).rolling(roll_n).sum()
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))

   
def cal_alpha004(start_date, end_date, start_minute, roll_n, roll_m, path):
    """
    三个收盘价均值方差、交易量均值相关条件，满足为1，不满足为-1\n
    nan行数=roll_n + roll_m - 1
    """
    factor_name = 'alpha004_start{a}_roll{b}_roll{c}'.format(a=start_minute, b=roll_n, c=roll_m)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算因子
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'vol'] = twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        condition1 = ((res_df['close'].rolling(roll_m).mean() + res_df['close'].rolling(roll_m).std()) < res_df['close'].rolling(roll_n).mean())
        condition2 = (res_df['close'].rolling(roll_n).mean() < (res_df['close'].rolling(roll_m).mean() - res_df['close'].rolling(roll_m).std()))
        condition3 = (1 <= res_df['vol'] / res_df['vol'].rolling(roll_n + roll_m).mean())

        condition = condition1 & condition2 & condition3
        condition = condition.astype(int)
        condition = condition.replace({0: -1})

        res_df[factor_name] = condition
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha005(start_date, end_date, start_minute, roll_n, path):
    """
    成交量和最高价时序排名的相关系数的区间最大值\n
    nan行数=roll_n - 1 + roll_n - 1
    """
    factor_name = 'alpha005_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算因子
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'vol'] = twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['vol_rank'] = res_df['vol'].rolling(roll_n).rank(pct=True)
        res_df['high_rank'] = res_df['high'].rolling(roll_n).rank(pct=True)
        res_df[factor_name] = res_df['high_rank'].rolling(roll_n).corr(res_df['vol_rank'])
        res_df[factor_name] = res_df[factor_name].replace([np.nan, np.inf, -np.inf], 0)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))

   
def cal_alpha006(start_date, end_date, start_minute, diff_n, path):
    """
    对开盘和最高价之和的差分值sign后的排名\n
    nan行数=diff_n
    """
    factor_name = 'alpha006_start{a}_diff{b}'.format(a=start_minute, b=diff_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算因子
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df[factor_name] = np.nan
        res_df['condition'] = (0.85 * res_df['open'] + 0.15 * res_df['high']).diff(diff_n).dropna()
        res_df.loc[res_df['condition'] > 0, factor_name] = 1
        res_df.loc[res_df['condition'] == 0, factor_name] = 0
        res_df.loc[res_df['condition'] < 0, factor_name] = -1

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha007(start_date, end_date, start_minute, diff_n, path):
    """
    vwap与收盘价的差值和3比较取最大最小值后的排名之和与成交量差分值排名的乘积\n
    nan行数=diff_n
    """
    factor_name = 'alpha007_start{a}_diff{b}'.format(a=start_minute, b=diff_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算因子
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()
            
            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        part1 = (np.maximum(res_df['vwap'] - res_df['close'], 3)).rank(pct=True)
        part2 = (np.minimum(res_df['vwap'] - res_df['close'], 3)).rank(pct=True)
        part3 = (res_df['vol'].diff(diff_n)).rank(pct=True)
        res_df[factor_name] = part1 + part2 * part3

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))

  
def cal_alpha008(start_date, end_date, start_minute, diff_n, path):
    """
    高低均值与vwap加权求和后的差分值的排名\n
    nan行数=diff_n
    """
    factor_name = 'alpha008_start{a}_diff{b}'.format(a=start_minute, b=diff_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算因子
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()
            
            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df[factor_name] = 0.1 * res_df['low'] + 0.1 * res_df['high'] + 0.8 * res_df['vwap']
        res_df[factor_name] = -res_df[factor_name].diff(diff_n).rank(pct=True)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha009(start_date, end_date, start_minute, ewm_a, path):
    """
    最高最低价与成交量比值的移动平均值\n
    nan行数=1
    """
    factor_name = 'alpha009_start{a}_alpha{b}'.format(a=start_minute, b=round(ewm_a, 2))
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算因子
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()
            
            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df[factor_name] = (res_df['high'] + res_df['low']) / 2 - ((res_df['high'].shift() + res_df['low'].shift()) / 2) * ((res_df['high'] - res_df['low']) / res_df['vol'])
        res_df[factor_name] = res_df[factor_name].ewm(alpha=ewm_a, adjust=False).mean()

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha010(start_date, end_date, start_minute, roll_n, path):
    """
    收益率标准差或收盘价平方与5比较的最大值的排名\n
    nan行数=roll_n - 1
    """
    factor_name = 'alpha010_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change()
        condtion = (res_df['return'] < 0)
        res_df['part1'] = np.where(condtion, res_df['return'].rolling(roll_n).std(), 0)
        res_df['part2'] = np.where(~condtion, res_df['close'], 0)
        res_df[factor_name] = (np.maximum((res_df['part1'] + res_df['part2']) ** 2, 5)).rank(pct=True)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])
        res_df['date'] = date_

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha011(start_date, end_date, start_minute, roll_n, path):
    """
    最高低收盘比值与成交量乘积的和\n
    nan行数=roll_n - 1(上午下午都有)
    """
    factor_name = 'alpha011_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        temp = ((res_df['close'] - res_df['low']) - (res_df['high'] - res_df['close'])) / (res_df['high'] - res_df['low'])
        res_df[factor_name] = (temp * res_df['vol']).rolling(roll_n).sum()
        res_df[factor_name] = res_df[factor_name].fillna(0)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])
        res_df['date'] = date_

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha012(start_date, end_date, start_minute, roll_n, path):
    """
    收开盘价与vwap差值的排名的乘积\n
    nan行数=roll_n - 1
    """
    factor_name = 'alpha012_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        part1 = (res_df['open'] - res_df['vwap'].rolling(roll_n).mean()).rank(pct=True)
        part2 = -(res_df['close'] - res_df['vwap']).abs().rank(pct=True)
        res_df[factor_name] = part1 * part2

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha013(start_date, end_date, start_minute, path):
    """
    高低价乘积开方与vwap的差值\n
    nan行数=0
    """
    factor_name = 'alpha013_start{a}'.format(a=start_minute)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)
    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        res_df[factor_name] = (((res_df['high'] - res_df['low']) ** 0.5) - res_df['vwap'])

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])
        res_df['date'] = date_

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha014(start_date, end_date, start_minute, shift_n, path):
    """
    当日收盘价与前五日收盘价的差值\n
    nan行数=shift_n
    """
    factor_name = 'alpha014_start{a}_shift{b}'.format(a=start_minute, b=shift_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        res_df[factor_name] = (res_df['close'] - res_df['close'].shift(shift_n))

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha015(start_date, end_date, start_minute, path):
    """
    当日开盘价与昨日收盘价比值-1\n
    nan行数=1
    """
    factor_name = 'alpha015_start{a}'.format(a=start_minute)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            twap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            twap = pd.DataFrame(twap, columns=['twap'])
            twap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = twap['twap'].max()
            res.loc[min_, 'low'] = twap['twap'].min()
            res.loc[min_, 'close'] = twap['twap'].iloc[-1]
            res.loc[min_, 'open'] = twap['twap'].iloc[0]
            res.loc[min_, 'vol'] = twap['vol'].sum()
            res.loc[min_, 'vwap'] = (twap['vol'] * twap['twap']).sum() / twap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅
        res_df[factor_name] = (res_df['open'] / res_df['close'].shift() - 1)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha016(start_date, end_date, start_minute, roll_n, path):
    """
    -（成交量排名与vwap排名相关系数的排名的区间最大值）\n
    nan行数=roll_n - 1 + roll_n - 1
    """
    factor_name = 'alpha016_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        temp1 = res_df['vol'].rank(pct=True)
        temp2 = res_df['vwap'].rank(pct=True)
        part = temp1.rolling(roll_n).corr(temp2)
        res_df[factor_name] = -part.rolling(roll_n).max()

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha017(start_date, end_date, start_minute, roll_n, diff_n, path):
    """
    vwap与15差值的排名乘收盘价差分值的次方\n
    nan行数=max(roll_n - 1, diff_n)
    """
    factor_name = 'alpha017_start{a}_roll{b}_diff{c}'.format(a=start_minute, b=roll_n, c=diff_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time < dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        temp1 = res_df['vwap'].rolling(roll_n).max()
        temp2 = res_df['close'] - temp1
        part1 = temp2.rank(pct=True)
        part2 = res_df['close'].diff(diff_n)
        res_df[factor_name] = part1 ** part2 
        res_df[factor_name] = res_df[factor_name].replace([np.inf], 0)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha018(start_date, end_date, start_minute, shift_n, path):
    """
    当日收盘价与前五日收盘价的比值\n
    nan行数=shift_n
    """
    factor_name = 'alpha018_start{a}_shift{b}'.format(a=start_minute, b=shift_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        delay5 = res_df['close'].shift(shift_n)
        res_df[factor_name] = res_df['close'] / delay5

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha019(start_date, end_date, start_minute, shift_n, path):
    """
    当日收盘价与前五日收盘价差值与收盘价或差值的比\n
    nan行数=shift_n
    """
    factor_name = 'alpha019_start{a}_shift{b}'.format(a=start_minute, b=shift_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        delay5 = res_df['close'].shift(shift_n)
        condition1 = (res_df['close'] < delay5)
        condition3 = (res_df['close'] > delay5)

        part1 = (np.where(condition1, res_df['close'], 0) - np.where(condition1, delay5, 0)) / np.where(condition1, delay5, np.inf)
        part2 = (np.where(condition3, res_df['close'], 0) - np.where(condition3, delay5, 0)) / np.where(condition3, delay5, np.inf)
        res_df[factor_name] = part1 + part2

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha020(start_date, end_date, start_minute, shift_n, path):
    """（
    当日收盘价与前六日收盘价的比值-1）*100\n
    nan行数=shift_n
    """
    factor_name = 'alpha020_start{a}_shift{b}'.format(a=start_minute, b=shift_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        delay6 = res_df['close'].shift(shift_n)
        res_df[factor_name] = (res_df['close'] - delay6) * 100 / delay6

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def rolling_regression(x, n):
    '''计算回归项系数'''
    x = x.reshape(1, -1).T
    y = np.arange(1, n + 1)
    beta = np.linalg.inv(x.T @ x) @ x.T @ y
    beta = beta[0]
    return beta


def cal_alpha021(start_date, end_date, start_minute, roll_n, path):
    """
    收盘价均值对数字序列回归的beta值\n
    nan行数=roll_n - 1 + roll_n - 1
    """
    factor_name = 'alpha021_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        A = res_df['close'].rolling(roll_n).mean()
        res_df[factor_name] = A.rolling(roll_n).apply(lambda x: rolling_regression(x, roll_n), raw=True)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha022(start_date, end_date, start_minute, roll_n, shift_n, ewm_a, path):
    """
    当日与前六日收盘价与收盘价均值比值的移动平均值\n
    nan行数=roll_n - 1 + shift_n + 1
    """
    factor_name = 'alpha022_start{a}_roll{b}_shift{c}_ewm{d}'.format(a=start_minute, b=roll_n, c=shift_n, d=round(ewm_a, 2))
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        part1 = (res_df['close'] - res_df['close'].rolling(roll_n).mean()) / res_df['close'].rolling(roll_n).mean()
        res_df[factor_name] = part1 - part1.shift(shift_n)
        res_df[factor_name] = res_df[factor_name].ewm(alpha=ewm_a, adjust=False).mean()

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha023(start_date, end_date, start_minute, roll_n, ewm_a, path):
    """
    收盘价标准差比值的移动平均值\n
    nan行数=roll_n - 1 + 1
    """
    factor_name = 'alpha023_start{a}_roll{b}_ewm{c}'.format(a=start_minute, b=roll_n, c=round(ewm_a, 2))
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        condition1 = (res_df['close'] > res_df['close'].shift())
        res_df['temp1'] = np.where(condition1, res_df['close'].rolling(roll_n).std(), 0)
        res_df['temp2'] = np.where(~condition1, res_df['close'].rolling(roll_n).std(), 0)
        part1 = res_df['temp1'].ewm(alpha=ewm_a, adjust=False).mean()
        part2 = res_df['temp2'].ewm(alpha=ewm_a, adjust=False).mean()
        res_df[factor_name] = part1 * 100 / (part1 + part2)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha024(start_date, end_date, start_minute, shift_n, ewm_a, path):
    """
    收盘价标准差比值的移动平均值\n
    nan行数=shift_n + 1
    """
    factor_name = 'alpha024_start{a}_shift{b}_ewm{c}'.format(a=start_minute, b=shift_n, c=round(ewm_a, 2))
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        delay5 = res_df['close'].shift(shift_n)
        res_df[factor_name] = (res_df['close'] - delay5).ewm(alpha=ewm_a, adjust=False).mean()

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def rolling_wma(x, y):
    return np.sum(x * y)


def cal_alpha025(start_date, end_date, start_minute, path):
    """
    -（收盘价差值排名与成交量时序线性衰减之和排名与收益率之和排名的乘积）\n
    ！！！内部超参没调！！！\n
    nan行数=max(10, (8-1)+(6-1))
    """    
    factor_name = 'alpha025_start{a}'.format(a=start_minute)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        part1 = (res_df['close'].diff(5)).rank(pct=True)

        n = 6
        temp = res_df['vol'] / res_df['vol'].rolling(8).mean()
        seq = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        weight = np.array(seq)
        temp1 = (temp.rolling(n).apply(lambda x: rolling_wma(x, weight), raw=True)).rank(pct=True)
        part2 = 1 - temp1
        rank_sum_ret = res_df['return'].rolling(10).sum().rank(pct=True)
        part3 = 1 + rank_sum_ret
        res_df[factor_name] = -part1 * part2 * part3

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha026(start_date, end_date, start_minute, corr_n, roll_n, shift_n, path):
    """
    时序收盘均值与收盘的差+vwap与收盘价相关系数\n
    shift_n: 单位为秒, <= corr_n * 60\n
    nan行数=max(roll_n-1, corr_n-1)
    """
    factor_name = 'alpha026_start{a}_corr{d}_roll{c}_shift{b}'.format(a=start_minute, b=shift_n, d=corr_n, c=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['vwap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算每分钟数据
            res = pd.DataFrame()
            res.loc[min_, 'high'] = vwap['vwap'].max()
            res.loc[min_, 'low'] = vwap['vwap'].min()
            res.loc[min_, 'close'] = vwap['vwap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['vwap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['vwap']).sum() / vwap['vol'].sum()

            if i >= corr_n:

                period_data = pd.concat(mindata_list[i - corr_n + 1: i + 1])

                # 每秒数据合并
                period_vwap = period_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
                period_vwap = pd.DataFrame(period_vwap, columns=['vwap'])
                period_vwap['delay'] = period_vwap['vwap'].shift(shift_n)

                # 计算因子
                res.loc[min_, 'part2'] = period_vwap[['vwap', 'delay']].corr().iloc[0, 1]
            
            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)
        res_df = res_df.fillna(0)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        part1 = res_df['close'].rolling(roll_n).mean() - res_df['close']
        res_df[factor_name] = part1 + res_df['part2']

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha027(start_date, end_date, start_minute, roll_n, path):
    """
    ！！！修改！！！\n
    收盘价与前3、6天收盘价之比的和的加权移动平均\n
    nan行数=roll_n*2 + roll_n*3 - 1 
    """
    factor_name = 'alpha027_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        delay1 = res_df['close'].shift(roll_n)
        delay2 = res_df['close'].shift(roll_n * 2)

        temp1 = (res_df['close'] - delay1) / delay1 * 100 + (res_df['close'] - delay2) / delay2 * 100
        weight = np.array([0.9 * i for i in range(roll_n * 3)])
        weight = weight / np.sum(weight)
        res_df[factor_name] = temp1.rolling(roll_n * 3).apply(lambda x: rolling_wma(x, weight), raw=True)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha028(start_date, end_date, start_minute, roll_n, ewm_a, path):
    """
    （收盘价-最低价/最高价-最低价）的移动平均-（收盘价-最低价/最高价-最低价）移动平均的移动平均\n
    nan行数=roll_n - 1 + 1
    """
    factor_name = 'alpha028_start{a}_roll{b}_ewm{c}'.format(a=start_minute, b=roll_n, c=round(ewm_a, 2))
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        temp1 = res_df['close'] - res_df['low'].rolling(roll_n).min()
        temp2 = res_df['high'].rolling(roll_n).max() - res_df['low'].rolling(roll_n).min()
        part1 = 3 * (temp1 * 100 / temp2).ewm(alpha=ewm_a, adjust=False).mean()
        temp3 = (temp1 * 100 / temp2).ewm(alpha=ewm_a, adjust=False).mean()
        part2 = 2 * temp3.ewm(alpha=ewm_a, adjust=False).mean()
        res_df[factor_name] = part1 - part2

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha029(start_date, end_date, start_minute, shift_n, path):
    """
    （当日收盘价与前六日收盘价比值-1）*成交量\n
    nan行数=shift_n
    """
    factor_name = 'alpha029_start{a}_shift{b}'.format(a=start_minute, b=shift_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        delay = res_df['close'].shift(shift_n)
        res_df[factor_name] = (res_df['close'] - delay) * res_df['vol'] / delay

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha030(start_date, end_date, start_minute, shift_n, path):
    """
    需要MKT,SMB,HML数据\n
    """
    return 0


def cal_alpha031(start_date, end_date, start_minute, roll_n, path):
    """
    收盘价与时序收盘均值的比值-1\n
    nan行数=roll_n - 1
    """
    factor_name = 'alpha031_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        res_df[factor_name] = (res_df['close'] - res_df['close'].rolling(roll_n).mean()) * 100 / res_df['close'].rolling(roll_n).mean()
        
        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha032(start_date, end_date, start_minute, roll_n, path):
    """
    -（最高价和成交量的排名的相关系数的排名之和）\n
    nan行数=roll_n - 1 + roll_n - 1
    """
    factor_name = 'alpha032_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅

        temp1 = res_df['high'].rank(pct=True)
        temp2 = res_df['vol'].rank(pct=True)
        temp3 = temp1.rolling(roll_n).corr(temp2)
        res_df[factor_name] = (temp3.rank(pct=True)).rolling(roll_n).sum()
        res_df[factor_name] = -res_df[factor_name]

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha033(start_date, end_date, start_minute, roll_n, roll_m, path):
    """
    当日和前五日的最低价时序最小值之差*前一年收益率均值*当日成交量的时序排名\n
    roll_m: 单位为秒, roll_m < roll_n * 60 \n
    nan行数=roll_n - 1 + roll_n
    """
    factor_name = 'alpha033_start{a}_roll{b}_roll{c}'.format(a=start_minute, b=roll_n, c=roll_m)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            min_period = roll_m // 60
            if i >= min_period:
                
                period_data = min_data[i - min_period: i + 1]

                # 每秒数据合并
                period_vwap = period_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
                period_vwap = pd.DataFrame(period_vwap, columns=['vwap'])

                period_vwap['return'] = period_vwap['vwap'].pct_change()
                period_vwap = period_vwap.iloc[-roll_m: ]
                res.loc[min_, 'return_ms'] = period_vwap['return'].sum()
                
            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅
        temp1 = res_df['low'].rolling(roll_n).min()
        part1 = temp1.shift(roll_n) - temp1
        temp2 = (res_df['return'].rolling(roll_n).sum() - res_df['return_ms']) / (roll_n * 60 - roll_m) # 20天前的220涨跌幅做日化
        part2 = temp2.rank(pct=True)
        temp3 = res_df['vol']
        part3 = temp3.rolling(roll_n).rank(pct=True)  # TS_RANK
        res_df[factor_name] = part1 * part2 * part3

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha034(start_date, end_date, start_minute, roll_n, path):
    """
    前12日收盘价均值/当日收盘价\n
    nan行数=roll_n - 1
    """
    factor_name = 'alpha034_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change() # 每分钟涨跌幅
        res_df[factor_name] = res_df['close'].rolling(roll_n).mean() / res_df['close']

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha035(start_date, end_date, start_minute, roll_n, roll_m, path):
    """
    min（收盘价差分值的时序线性衰减之和的排名，成交量与开盘价相关系数的时序线性衰减之和的排名）\n
    nan行数=max(roll_n - 1, 2*(roll_m - 1))
    """
    factor_name = 'alpha035_start{a}_roll{b}_roll{c}'.format(a=start_minute, b=roll_n, c=roll_m)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        seq1 = [2 * i / (roll_n * (roll_n + 1)) for i in range(1, roll_n + 1)]
        seq2 = [2 * i / (roll_m * (roll_m + 1)) for i in range(1, roll_m + 1)]
        weight1 = np.array(seq1)
        weight2 = np.array(seq2)

        temp1 = res_df['open'].diff()
        part1 = temp1.rolling(roll_n).apply(lambda x: rolling_wma(x, weight1), raw=True)
        part1 = part1.rank(pct=True)

        temp2 = res_df['open'].rolling(roll_m).corr(res_df['vol'])
        temp2 = temp2.replace([np.nan, np.inf, -np.inf], 0)
        part2 = temp2.rolling(roll_m).apply(lambda x: rolling_wma(x, weight2), raw=True)

        res_df[factor_name] = np.minimum(part1, part2)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha036(start_date, end_date, start_minute, roll_n, roll_m, path):
    """
    成交量与vwap排名的相关系数之和的排名\n
    nan行数=roll_n - 1 + roll_m - 1
    """
    factor_name = 'alpha036_start{a}_roll{b}_roll{c}'.format(a=start_minute, b=roll_n, c=roll_m)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        temp1 = res_df['vol'].rank(pct=True)
        temp2 = res_df['vwap'].rank(pct=True)

        part1 = temp1.rolling(roll_n).corr(temp2)
        part1 = part1.replace([np.nan, np.inf, -np.inf], 0)
        res_df[factor_name] = part1.rolling(roll_m).sum().rank(pct=True)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha037(start_date, end_date, start_minute, roll_n, shift_n, path):
    """
    -【当日的（5日内开盘价之和*5日内收益率之和）-前十天的（5日内开盘价之和*5日内收益率之和）】的排名\n
    nan行数=roll_n - 1 + shift_n
    """
    factor_name = 'alpha037_start{a}_roll{b}_shift{c}'.format(a=start_minute, b=roll_n, c=shift_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        res_df['return'] = res_df['close'].pct_change()
        temp = res_df['open'].rolling(roll_n).sum() * res_df['return'].rolling(roll_n).sum()
        part2 = temp.shift(shift_n)
        res_df[factor_name]  = -(temp - part2).rank(pct=True)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha038(start_date, end_date, start_minute, roll_n, path):
    """
    若20内最高价均值小于当日最高价，值为-（最高价2日差分值），否则为0\n
    nan行数=roll_n - 1 + 2
    """
    factor_name = 'alpha038_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        MA_n = res_df['high'].rolling(roll_n).mean()
        delta2 = res_df['high'].diff(2)
        condition = (MA_n < res_df['high'])
        res_df[factor_name] = np.where(condition, -delta2, 0)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


def cal_alpha039(start_date, end_date, start_minute, roll_n, corr_n, path):
    """
    -（收盘价差分值的时序线性衰减之和的排名-价格与成交量相关系数的时序线性衰减之和的排名）\n
    nan行数=roll_n - 1 + 2
    """
    factor_name = 'alpha039_start{a}_roll{b}'.format(a=start_minute, b=roll_n)
    file_list = os.listdir(fc.kline_data)
    file_dates = pd.Series(pd.to_datetime(n_[8: 16], format="%Y%m%d") for n_ in file_list).drop_duplicates()
    file_dates = file_dates[(file_dates > start_date) & (file_dates <= end_date)]
    file_dates = file_dates.reset_index(drop=True)

    result_list = []

    for date_ in tqdm(file_dates, desc=factor_name):
        
        # 读取当日主力数据
        data_path = read_file(date_)
        data = pd.read_pickle(os.path.join(fc.kline_data, data_path))[['price', 'date', 'vol']]

        # 处理时间
        data['date'] = data.apply(lambda x: pt.DoubleToDatetime(x['date']), axis=1)
        data['amount'] = data['price'] * data['vol']
        data['minute'] = data['date'].dt.floor('T')

        data = data[data['vol'] != 0]

        # 舍弃集合竞价与收盘数据
        data = data[data['minute'].dt.time >= dt.time(9, 30)]
        data = data[data['minute'].dt.time <= dt.time(15, 00)]
        data = data[data['minute'].dt.time != dt.time(11, 30)]

        # 按分钟分块
        min_list = list(data['minute'].unique())
        mindata_list = [data[data['minute'] == min_] for min_ in min_list]

        # 保存结果
        res_list = list()

        # 计算分钟数据
        for i in range(len(min_list)):
            
            min_ = min_list[i]
            min_data = mindata_list[i]

            # 每秒数据合并
            vwap = min_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
            vwap = pd.DataFrame(vwap, columns=['twap'])
            vwap['vol'] = min_data.groupby('date')['vol'].sum()

            # 计算因子
            res = pd.DataFrame(columns=['high', 'low', 'close'])
            res.loc[min_, 'high'] = vwap['twap'].max()
            res.loc[min_, 'low'] = vwap['twap'].min()
            res.loc[min_, 'close'] = vwap['twap'].iloc[-1]
            res.loc[min_, 'open'] = vwap['twap'].iloc[0]
            res.loc[min_, 'vol'] = vwap['vol'].sum()
            res.loc[min_, 'vwap'] = (vwap['vol'] * vwap['twap']).sum() / vwap['vol'].sum()

            if i >= (corr_n // 60 + 1):

                period_data = pd.concat(mindata_list[i - corr_n + 1: i + 1])

                # 每秒数据合并
                period_vwap = period_data.groupby('date').apply(lambda x: x['amount'].sum() / x['vol'].sum())
                period_vwap = pd.DataFrame(period_vwap, columns=['vwap'])
                

                # 计算因子
                res.loc[min_, 'part2'] = period_vwap[['vwap', 'delay']].corr().iloc[0, 1]
            
            # 保存每分钟结果
            res_list.append(res)

        # 保存每日结果
        res_df = pd.concat(res_list)

        # 计算因子
        MA_n = res_df['high'].rolling(roll_n).mean()
        delta2 = res_df['high'].diff(2)
        condition = (MA_n < res_df['high'])
        res_df[factor_name] = np.where(condition, -delta2, 0)

        # 设置时间
        res_df = res_df.reset_index(names=['minute'])

        # 保存每日结果
        result_list.append(res_df)

    # 保存时间段因子结果
    result_df = pd.concat(result_list)

    # 加入起始时间
    start_minute = dt.timedelta(minutes=start_minute)
    start_time = dt.datetime.combine(dt.datetime.today().date(), dt.time(9, 30))
    start_time = start_time + start_minute
    start_time = start_time.time()
    result_df['time'] = result_df['minute'].dt.time
    result_df = result_df[result_df['time'] >= start_time]

    result_df[['minute', factor_name]].to_pickle(os.path.join(path, '%s.pkl' % factor_name))


# %%
if __name__ == '__main__':

    cal_alpha001(fc.start_date, fc.end_date, start_minute=fc.start_minute, corr_n=10, path=fc.factor_save_path)

    cal_alpha002(fc.start_date, fc.end_date, start_minute=fc.start_minute, path=fc.factor_save_path)

    cal_alpha003(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=10, path=fc.factor_save_path)

    cal_alpha004(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=2, roll_m=8, path=fc.factor_save_path)

    cal_alpha005(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=5, path=fc.factor_save_path)

    cal_alpha006(fc.start_date, fc.end_date, start_minute=fc.start_minute, diff_n=5, path=fc.factor_save_path)

    cal_alpha007(fc.start_date, fc.end_date, start_minute=fc.start_minute, diff_n=5, path=fc.factor_save_path)

    cal_alpha008(fc.start_date, fc.end_date, start_minute=fc.start_minute, diff_n=5, path=fc.factor_save_path)

    cal_alpha009(fc.start_date, fc.end_date, start_minute=fc.start_minute, ewm_a=2.0 / 7, path=fc.factor_save_path)

    cal_alpha010(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=10, path=fc.factor_save_path)

    cal_alpha011(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=10, path=fc.factor_save_path)

    cal_alpha012(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=10, path=fc.factor_save_path)

    cal_alpha013(fc.start_date, fc.end_date, start_minute=fc.start_minute, path=fc.factor_save_path)

    cal_alpha014(fc.start_date, fc.end_date, start_minute=fc.start_minute, shift_n=10, path=fc.factor_save_path)

    cal_alpha015(fc.start_date, fc.end_date, start_minute=fc.start_minute, path=fc.factor_save_path)

    cal_alpha016(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=6, path=fc.factor_save_path)

    cal_alpha017(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=10, diff_n=5, path=fc.factor_save_path)

    cal_alpha018(fc.start_date, fc.end_date, start_minute=fc.start_minute, shift_n=5, path=fc.factor_save_path)

    cal_alpha019(fc.start_date, fc.end_date, start_minute=fc.start_minute, shift_n=5, path=fc.factor_save_path)
    
    cal_alpha020(fc.start_date, fc.end_date, start_minute=fc.start_minute, shift_n=6, path=fc.factor_save_path)
    
    cal_alpha021(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=6, path=fc.factor_save_path)
    
    cal_alpha022(fc.start_date, fc.end_date, start_minute=fc.start_minute, shift_n=4, roll_n=6, ewm_a=1.0 / 12, path=fc.factor_save_path)
    
    cal_alpha023(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=6, ewm_a=1.0 / 20, path=fc.factor_save_path)
    
    cal_alpha024(fc.start_date, fc.end_date, start_minute=fc.start_minute, shift_n=5, ewm_a=1.0 / 12, path=fc.factor_save_path)
    
    cal_alpha025(fc.start_date, fc.end_date, start_minute=fc.start_minute, path=fc.factor_save_path)
   
    cal_alpha026(fc.start_date, fc.end_date, start_minute=fc.start_minute, corr_n=4, shift_n=20, roll_n=6, path=fc.factor_save_path)
   
    cal_alpha027(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=2, path=fc.factor_save_path)
    
    cal_alpha028(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=6, ewm_a=1.0 / 3, path=fc.factor_save_path)
    
    cal_alpha029(fc.start_date, fc.end_date, start_minute=fc.start_minute, shift_n=5, path=fc.factor_save_path)
    
    cal_alpha031(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=6, path=fc.factor_save_path)
    
    cal_alpha032(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=6, path=fc.factor_save_path)
    
    cal_alpha033(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=5, roll_m=240, path=fc.factor_save_path)
    
    cal_alpha034(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=10, path=fc.factor_save_path)

    cal_alpha035(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=10, roll_m=5, path=fc.factor_save_path)
    
    cal_alpha036(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=10, roll_m=5, path=fc.factor_save_path)
    
    cal_alpha037(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=6, shift_n=5, path=fc.factor_save_path)

    cal_alpha038(fc.start_date, fc.end_date, start_minute=fc.start_minute, roll_n=8, path=fc.factor_save_path)
    
