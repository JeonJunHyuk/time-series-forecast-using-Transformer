from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import gc
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyunpack
import wget

from configs import ExperimentConfig

def download_from_url(url, output_path):
    print('Pulling data from {} to {}'.format(url, output_path))
    wget.download(url,output_path)
    print('done')


def recreate_folder(path):
    shutil.rmtree(path)
    os.makedirs(path)


def unzip(zip_path, output_file, data_folder):
    print('unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    if not os.path.exists(output_file):
        raise ValueError('Error in unzipping process! {} not found.'.format(output_file))


def download_and_unzip(url, zip_path, csv_path, data_folder):
    download_from_url(url, zip_path)
    unzip(zip_path, csv_path, data_folder)
    print('Done.')

def process_nike(config):
    data_folder = config.data_folder

    df = pd.read_excel(os.path.join(data_folder, 'nike_data.xlsx'), index_col=0)
    df = df.iloc[:, 0:3]
    df = df[:-2]  # 마지막 두 줄 0이라..
    df = df[df['cAOV'].notnull()]
    df = np.log(df)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df['date'] = df.index
    df['days_from_start'] = (df.index - df.index[0]).days
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.weekofyear
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['ID'] = 'single'     # ID 지금은 없지만.. 넣어야해.
    df['static'] = 'static'  # static이 없어..

    event_days = pd.read_csv(os.path.join(data_folder,'nike_launch_date.csv',))
    event_days['type'] = 'event'
    event_days['launch_date'] = pd.to_datetime(event_days['launch_date'], format='%Y%m%d')

    df['event_day'] = df.merge(event_days, left_on=['date'], right_on=['launch_date'], how='left')['type'].fillna('')
    # df.reset_index(drop=True, inplace=True)

    output_file = config.data_csv_path
    print('Completed formatting, saving to {}'.format(output_file))
    df.to_csv(output_file)

    print('Done.')

def process_nike_traffic(config):
    data_folder = config.data_folder

    traffic = pd.read_csv(os.path.join(data_folder, 'nike_traffic.csv'))
    traffic = traffic[
        (traffic['Store Nm'] == 'All platform') & (traffic['Lob Cd'].isnull()) & (traffic['Order Fg'] == 'DM')]
    traffic = traffic.drop(['Store Nm', 'Lob Cd', 'Order Fg'], axis=1)
    traffic = traffic.rename(columns={'Month, Day, Year of Order_Date': 'date'})
    traffic['Traffic'] = traffic['Traffic'].str.replace(',', '').astype('int64')
    traffic['Units'] = traffic['Units'].str.replace(',', '').astype('int64')
    traffic['AMOUNT_KRW'] = traffic['AMOUNT_KRW'].str.replace(',', '').astype('int64')
    traffic['Transactions'] = traffic['Transactions'].str.replace(',', '').astype('int64')
    traffic = traffic[traffic['Traffic'] > 0]
    traffic['date'] = pd.to_datetime(traffic['date'], format='%Y년 %m월 %d일')

    traffic['Transactions'] = np.log(traffic['Transaction'])
    traffic['Amount_KRW'] = np.log(traffic['Amount KRW'])

    traffic.index = traffic['date']
    traffic['days_from_start'] = (traffic.index - traffic.index[0]).days
    traffic['day_of_week'] = traffic.index.dayofweek
    traffic['day_of_month'] = traffic.index.day
    traffic['week_of_year'] = traffic.index.weekofyear
    traffic['month'] = traffic.index.month
    traffic['year'] = traffic.index.year

    traffic['ID'] = 'traffic'
    traffic['static'] = 'traffic'

    launch_days = pd.read_csv(os.path.join(data_folder, 'nike_launch_date.csv', ))
    launch_days['yn'] = 'yes'
    launch_days['launch_date'] = pd.to_datetime(launch_days['launch_date'], format='%Y%m%d')
    traffic['launch'] = traffic.merge(launch_days, left_on=['date'], right_on=['launch_date'], how='left')['yn'].fillna('')

    event_days = pd.read_csv(os.path.join(data_folder, 'nike_event_date.csv', ))
    event_days['yn'] = 'yes'
    event_days['event_date'] = pd.to_datetime(event_days['event_date'], format='%Y%m%d')
    traffic['event'] = traffic.merge(launch_days, left_on=['date'], right_on=['event_date'], how='left')['yn'].fillna('')

    output_file = config.data_csv_path
    print('Completed formatting, saving to {}'.format(output_file))
    traffic.to_csv(output_file)

    print('Done.')



def process_nike_tran(config):
    data_folder = config.data_folder

    tran = pd.read_csv(os.path.join(data_folder,'nike_tran.csv'))
    tran['Transaction'] = tran['Transaction'].str.replace(',', '').astype('int64')
    tran = tran.rename(columns={'Month, Day, Year of Order Ymd': 'date'})
    tran = tran[tran['Transaction'] > 0]
    tran = tran[(tran['Order Fg'] == 'DM') | (tran['Order Fg'] == 'CN')]
    tran['Quantity'] = tran['Quantity'].str.replace(',', '').astype('int64')
    tran['FCST_QTY'] = tran['FCST_QTY'].fillna(0)
    tran['FCST_PER'] = tran['FCST_PER'].fillna(0)
    tran['FCST_QTY'] = tran['FCST_QTY'].apply(str)
    tran['FCST_QTY'] = tran['FCST_QTY'].str.replace(',', '').astype('int64')
    tran['FCST_PER'] = tran['FCST_PER'].apply(str)
    tran['FCST_PER'] = tran['FCST_PER'].str.replace(',', '').astype('int64')
    tran['Amount Krw'] = tran['Amount Krw'].str.replace(',', '').astype('int64')
    tran['FCST_QTY'] = tran['FCST_QTY'].replace(0, np.nan).fillna('')
    tran['FCST_PER'] = tran['FCST_PER'].replace(0, np.nan).fillna('')
    tran['date'] = pd.to_datetime(tran['date'], format='%Y년 %m월 %d일')

    tran['Transaction'] = np.log(tran['Transaction'])
    tran['Amount Krw'] = np.log(tran['Amount Krw'])

    tran.index = tran['date']
    tran.index.names = ['index_date']
    tran['days_from_start'] = (tran.index - tran.index[0]).days
    tran['day_of_week'] = tran.index.dayofweek
    tran['day_of_month'] = tran.index.day
    tran['week_of_year'] = tran.index.weekofyear
    tran['month'] = tran.index.month
    tran['year'] = tran.index.year

    tran['ID'] = tran['Order Fg']+'_'+tran['Lob Cd']

    launch_days = pd.read_csv(os.path.join(data_folder, 'nike_launch_date.csv', ))
    launch_days['yn'] = 'yes'
    launch_days['launch_date'] = pd.to_datetime(launch_days['launch_date'], format='%Y%m%d')
    tran['launch'] = tran.merge(launch_days, left_on=tran['date'], right_on=['launch_date'], how='left')['yn'].fillna('')

    event_days = pd.read_csv(os.path.join(data_folder, 'nike_event_date.csv', ))
    event_days['yn'] = 'yes'
    event_days['event_date'] = pd.to_datetime(event_days['event_date'], format='%Y%m%d')
    tran['event'] = tran.merge(event_days, left_on=tran['date'], right_on=['event_date'], how='left')['yn'].fillna('')

    output_file = config.data_csv_path
    print('Completed formatting, saving to {}'.format(output_file))
    tran.to_csv(output_file)

    print('Done.')

def process_nike_traffic_0706(config):
    # data_folder = config.data_folder
    data_folder = 'c:/users/junhyuk/nike_tft/0706_data'
    traffic = pd.read_csv(os.path.join(data_folder,'traffic_0706_S.csv'),encoding='utf-16',delimiter='\t')
    traffic.columns = ['year','month','day','store','traffic']
    traffic['year'] = traffic.year.str.split(' ', expand=True).iloc[:, 2]
    traffic['date'] = traffic['year'] + '-' + traffic['month'] + '-' + traffic['day'].astype('str')
    traffic['date'] = pd.to_datetime(traffic['date'], format='%Y-%m월-%d')
    traffic.drop(['year','month','day'], axis=1, inplace=True)
    traffic['traffic'] = traffic['traffic'].str.replace(',', '').astype('int64')
    # traffic snkrs 는 2019.6.28 부터 할 것
    transaction = pd.read_csv(os.path.join(data_folder,'transaction_SL.csv'),encoding='utf-16',delimiter='\t')
    transaction.columns = ['date', 'lob', 'store', 'transaction']
    transaction['date'] = pd.to_datetime(transaction['date'], format='%Y년 %m월 %d일')
    transaction['transaction'] = transaction['transaction'].str.replace(',', '').astype('int64')
    transaction = transaction.groupby(['date', 'store']).sum().reset_index()

    traffic = traffic.merge(transaction, on=['date','store'])

    dm = pd.read_csv(os.path.join(data_folder,'DM_amount_qty_SL.csv'),encoding='utf-16',delimiter='\t')
    dm.columns = ['date','lob','store','dm_amount','dm_qty']
    dm = dm[dm.dm_qty.notnull()]




def download_volatility(config):
    data_folder = config.data_folder
    csv_path = os.path.join(data_folder, 'oxfordmanrealizedvolatilityindices.csv')

    df = pd.read_csv(csv_path, index_col=0)
    idx = [str(s).split('+')[0] for s in df.index]
    dates = pd.to_datetime(idx)
    df['date'] = dates
    df['days_from_start'] = (dates - pd.datetime(2000,1,3)).days
    df['day_of_week'] = dates.dayofweek
    df['day_of_month'] = dates.day
    df['week_of_year'] = dates.weekofyear
    df['month'] = dates.month
    df['year'] = dates.year
    df['categorical_id'] = df['Symbol'].copy()

    vol = df['rv5_ss'].copy()
    vol.loc[vol==0.] = np.nan
    df['log_vol'] = np.log(vol)

    symbol_region_mapping = {
        '.AEX': 'EMEA',
        '.AORD': 'APAC',
        '.BFX': 'EMEA',
        '.BSESN': 'APAC',
        '.BVLG': 'EMEA',
        '.BVSP': 'AMER',
        '.DJI': 'AMER',
        '.FCHI': 'EMEA',
        '.FTMIB': 'EMEA',
        '.FTSE': 'EMEA',
        '.GDAXI': 'EMEA',
        '.GSPTSE': 'AMER',
        '.HSI': 'APAC',
        '.IBEX': 'EMEA',
        '.IXIC': 'AMER',
        '.KS11': 'APAC',
        '.KSE': 'APAC',
        '.MXX': 'AMER',
        '.N225': 'APAC ',
        '.NSEI': 'APAC',
        '.OMXC20': 'EMEA',
        '.OMXHPI': 'EMEA',
        '.OMXSPI': 'EMEA',
        '.OSEAX': 'EMEA',
        '.RUT': 'EMEA',
        '.SMSI': 'EMEA',
        '.SPX': 'AMER',
        '.SSEC': 'APAC',
        '.SSMI': 'EMEA',
        '.STI': 'APAC',
        '.STOXX50E': 'EMEA'
    }

    df['Region'] = df['Symbol'].apply(lambda k: symbol_region_mapping[k])

    output_df_list = []
    for grp in df.groupby('Symbol'):
        sliced = grp[1].copy()
        sliced.sort_values('days_from_start', inplace=True)
        sliced['log_vol'].fillna(method='ffill', inplace=True)
        sliced.dropna()
        output_df_list.append(sliced)

    df = pd.concat(output_df_list, axis=0)

    output_file = config.data_csv_path
    print('Completed formatting, saving to {}'.format(output_file))
    df.to_csv(output_file)

    print('Done.')


def process_favorita(config):
    url = 'http://www.kaggle.com/c/favorita-crocery-sales-forecasting/data'
    data_folder = config.data_folder
    zip_file = os.path.join(data_folder,'favorita-grocery-sales-forecasting.zip')
    # download_and_unzip(url, zip_file, data_folder ,data_folder)

    if not os.path.exists(zip_file):
        raise ValueError(
            'Favorita zip file not found in {}'.format(zip_file) +
            'Please manually download data from Kaggle @ {}'.format(url))

    outputs_file = os.path.join(data_folder, 'train.csv.7z')
    unzip(zip_file, outputs_file, data_folder)

    for file in glob.glob(os.path.join(data_folder, '*.7z')):
        csv_file = file.replace('.7z', '')
        unzip(file, csv_file, data_folder)
    print('Unzipping complete, commencing data processing...')

    start_date = pd.datetime(2015,1,1)   # pd.datetime 없어질 것. datetime.datetime 따로 쓰셈.
    end_date = pd.datetime(2016,6,1)

    print('Regenerating data...')

    temporal = pd.read_csv(os.path.join(data_folder,'train.csv'), index_col=0)
    store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
    # oil = pd.read_csv(os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[:, 0]
    holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
    items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
    transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

    temporal['date'] = pd.to_datetime(temporal['date'])

    if start_date is not None:
        temporal = temporal[(temporal['date'] >= start_date)]
    if end_date is not None:
        temporal = temporal[(temporal['date'] < end_date)]

    # dates = temporal['date'].unique()

    temporal['traj_id'] = temporal['store_nbr'].apply(str)+'_'+temporal['item_nbr'].apply(str)
    temporal['unique_id'] = temporal['traj_id']+'_'+temporal['date'].apply(str)

    print('Removing returns data')
    min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
    valid_ids = set(min_returns[min_returns >= 0].index)
    selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
    new_temporal = temporal[selector].copy()
    del temporal
    gc.collect()
    temporal = new_temporal
    temporal['open'] = 1

    print('Resampling to regular grid')
    resampled_dfs = []
    for traj_id, raw_sub_df in temporal.groupby('traj_id'):
        print('Resampling', traj_id)
        sub_df = raw_sub_df.set_index('date', drop=True).copy()
        sub_df = sub_df.resample('1d').last()
        sub_df['date'] = sub_df.index
        sub_df[['store_nbr', 'item_nbr', 'onpromotion']] = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].fillna(method='ffill')
        sub_df['open'] = sub_df['open'].fillna(0)
        sub_df['log_sales'] = np.log(sub_df['unit_sales'])
        resampled_dfs.append(sub_df.reset_index(drop=True))
    new_temporal = pd.concat(resampled_dfs, axis=0)
    del temporal
    gc.collect()
    temporal = new_temporal

    # temporal.to_csv('temp.csv')

    # print('Adding oil')
    # oil.name = 'oil'
    # # oil.index = pd.to_datetime(oil.index)
    # temporal = temporal.join(oil.fillna(method='ffill'), on='date', how='left')
    # temporal['oil'] = temporal['oil'].fillna(-1)

    print('Adding store info')
    temporal = temporal.join(store_info, on='store_nbr', how='left')

    print('Adding item info')
    temporal = temporal.join(items, on='item_nbr', how='left')

    transactions['date'] = pd.to_datetime(transactions['date'])
    temporal = temporal.merge(
        transactions,
        left_on=['date', 'store_nbr'],
        right_on=['date', 'store_nbr'],
        how='left')
    temporal['transactions'] = temporal['transactions'].fillna(-1)

    temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
    temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
    temporal['month'] = pd.to_datetime(temporal['date'].values).month

    print('Adding holidays')
    holiday_subset = holidays[holidays['transferred'].apply(lambda x: not x)].copy()
    holiday_subset.columns = [s if s != 'type' else 'holiday_type' for s in holiday_subset.columns]
    holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
    local_holidays = holiday_subset[holiday_subset['locale']=='Local']
    regional_holidays = holiday_subset[holiday_subset['locale']=='Regional']
    national_holidays = holiday_subset[holiday_subset['locale']=='National']

    temporal['national_hol'] = temporal.merge(
        national_holidays, left_on=['date'], right_on=['date'],
        how='left')['description'].fillna('')
    temporal['regional_hol'] = temporal.merge(
        regional_holidays, left_on=['state', 'date'],
        right_on=['locale_name', 'date'],
        how='left')['description'].fillna('')
    temporal['local_hol'] = temporal.merge(
        local_holidays,
        left_on=['city', 'date'],
        right_on=['locale_name', 'date'],
        how='left')['description'].fillna('')

    temporal.sort_values('unique_id', inplace=True)

    print('Saving processed file to {}'.format(config.data_csv_path))
    temporal.to_csv(config.data_csv_path)


def main(expt_name, force_download, output_folder):
    print('Running download script')
    expt_config = ExperimentConfig(expt_name, output_folder)
    if os.path.exists(expt_config.data_csv_path) and not force_download:
        print('Data has been processed for {}. Skipping download...'.format(expt_name))
        sys.exit(0)
    else:
        print('Resetting data folder...')
        # recreate_folder(expt_config.data_folder)

    download_functions = {
        'favorita': process_favorita,
        'volatility': download_volatility,
        'nike': process_nike,
        'nike_tran': process_nike_tran
    }

    if expt_name not in download_functions:
        raise ValueError('Unrecognized experiment. name={}'.format(expt_name))

    download_function = download_functions[expt_name]
    print('Getting {} data...'.format(expt_name))
    download_function(expt_config)

    print('Download & first preprocess completed.')


if __name__== '__main__':

    def get_args():
        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description='Data download configs')
        parser.add_argument(
            'expt_name',
            metavar='e',
            type=str,
            nargs='?',
            choices=experiment_names,
            help='Experiment Name. Defualt={}'.format(','.join(experiment_names)))
        parser.add_argument(
            'output_folder',
            metavar='f',
            type=str,
            nargs='?',
            default='.',
            help='Path to folder for data download')
        parser.add_argument(
            'force_download',
            metavar='r',
            type=str,
            nargs='?',
            choices=['yes','no'],
            default='no',
            help='Whether to re-run data download')

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == '.' else args.output_folder

        return args.expt_name, args.force_download=='yes', root_folder

    name, force, folder = get_args()
    main(expt_name=name, force_download=force, output_folder=folder)