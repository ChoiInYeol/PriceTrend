import os

import pandas as pd

import yfinance as yf
yf.pdr_override()

from pandas_datareader import data as pdr

from datetime import datetime
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import scienceplots
plt.style.use('science')
plt.rcParams.update({'figure.dpi': '100'})
# LaTeX 렌더링 비활성화 (LaTeX 설치 문제 회피)
plt.rcParams.update({
    "text.usetex": False
})
pd.options.mode.chained_assignment = None

# 경고 무시
import warnings
warnings.filterwarnings(action='ignore')

from fredapi import Fred
fred = Fred(api_key='92aee5bade30e7550ef9811af1ace8a2') # API Key ! ! !
 
def data_formatter(df):
    
    df.index = pd.to_datetime(df.index)
    df['DATE'] = df.index
    # @Debug: print(df.dtypes) # df.info()
    df['YEAR'] = df['DATE'].dt.strftime('%Y').astype(str)
    df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y%m').astype(str)
    df = df.reset_index(drop=True)

    return df


def get_price_data(tickers, start, end):
    df = pd.DataFrame(columns=tickers)

    for ticker in tickers:
        df[ticker] = pdr.get_data_yahoo(ticker, start, end)['Adj Close']

    df = data_formatter(df)
    
    return df


def get_unemployment(start, end):
    
    series_unemployment = fred.get_series('UNRATE', start, end)
    
    # df로 변경
    df_unemployment = pd.DataFrame(series_unemployment)
    df_unemployment = df_unemployment.rename(columns={0: 'UNEMPLOYMENT_RATE'})
    
    # 날짜를 datetime으로 변경
    df_unemployment.index = pd.to_datetime(df_unemployment.index)
    df_unemployment['UEM_CHANGE'] = df_unemployment['UNEMPLOYMENT_RATE'].pct_change()
    
    # @Tip: df_unemployment['UEM_CHANGE'] = df_unemployment['UNEMPLOYMENT_RATE'] - df_unemployment[
    # 'UNEMPLOYMENT_RATE'].shift(1)
    
    return df_unemployment


def select_dual_momentum(spy_m12_return, bil_m12_return, efa_m12_return):
    if spy_m12_return > bil_m12_return:
        if spy_m12_return > efa_m12_return:
            return 'SPY'
        else:
            return 'EFA'
    else:
        return 'AGG'


def select_qqq_or_shy(spy_index, spy_ma_200d, uem_index, uem_12m):
    if spy_index < spy_ma_200d and uem_index > uem_12m:
        return 'SHY'
    else:
        return 'QQQ'

def asset_data(start, end):
    tickers_laa = ['SPY', 'GLD', 'IEF', 'IWD', 'QQQ', 'SHY', 'SMH']
    tickers_dual_momentum = ['SPY', 'EFA', 'AGG', 'BIL']

    df_asset_laa = get_price_data(tickers_laa, start, end)
    
    # # DATE를 기준으로 df_asset_laa를 PriceTrendsPrice.csv와 merge
    # path = '/home/indi/codespace/PriceTrend/LAA/pf_data/PriceTrendsPrice2.csv'
    # PriceTrendsPrice = pd.read_csv(path)
    
    # # 인덱스를 datetime으로 변경
    # PriceTrendsPrice['DATE'] = pd.to_datetime(PriceTrendsPrice.index)
    # df_asset_laa = df_asset_laa.merge(PriceTrendsPrice, on='DATE', how='left')
    
    # df_asset_laa.to_csv('df_asset_laa.csv', index=False)
    
    df_asset_dual = get_price_data(tickers_dual_momentum, start, end)
    uem = get_unemployment(start, end)
    uem_monthly = data_formatter(uem)

    return df_asset_laa, df_asset_dual, uem_monthly


def laa_backtest(df_asset_laa : pd.DataFrame, uem_monthly : pd.DataFrame, my_etf_ticker : str = 'QQQ') -> pd.DataFrame:
    # LAA
    '''
    월말 리밸런싱을 가정.

    LAA
    고정자산: 미국 대형가치주 IWD, 금 GLD, 미국 중기국채 IEF
    타이밍 자산: 나스닥 QQQ, 미국 단기국채 SHY
    
    자산의 각 25%를 IWD, GLD, IEF에 투자
    나머지 25%는 QQQ 또는 SHY 에 투자.
    
    미국 S&P 500 지수 가격이 200일 이동평균보다 낮고 미국 실업률이 12개월 이동평균보다 높은 경우 SHY에 투자.
    그렇지 않을 경우 QQQ 투자
    
    이 때, QQQ 대신에 내가 임의로 만든 ETF로 대체할 수 있어야 함.
    해당 TICKER는 PRT이며, df_asset_laa에 추가되어 있음.
    
    uem_monthly는 미국 실업률 데이터를 담고 있음.
    
    연 1회 리밸런싱 : 고정자산
    월 1회 리밸런싱 : 타이밍 자산
    '''
    
    # @Tip: as_index=False -> Don't lose join column
    # 월말 및 연말 데이터 추출
    df_asset_laa_monthly = df_asset_laa.groupby('YEAR_MONTH', as_index=False).last().dropna()
    df_asset_laa_yearly = df_asset_laa_monthly.groupby('YEAR', as_index=False).last().dropna()

    df_spy = df_asset_laa[['SPY', 'DATE']]
    df_spy['MA_200D'] = df_spy.loc[:, 'SPY'].rolling(200).mean()
    df_spy = df_spy.dropna()

    df_uem = uem_monthly[['UNEMPLOYMENT_RATE', 'UEM_CHANGE', 'DATE', 'YEAR_MONTH', 'YEAR']]
    df_uem['MA_12M'] = df_uem.loc[:, 'UNEMPLOYMENT_RATE'].rolling(12).mean()
    df_uem = df_uem.dropna()

    rebalancing_yearly_dates = df_asset_laa_yearly['DATE'].tolist()
    rebalancing_monthly_dates = df_asset_laa_monthly['DATE'].tolist()
    rebalancing_monthly_spy_ma_200d = df_spy.loc[df_spy['DATE'].isin(rebalancing_monthly_dates)]

    begin_date_of_investment = rebalancing_monthly_dates[0]
    budget = 10000  # USD

    prev_cash_amount = 0
    prev_iwd_quantity = prev_gld_quantity = prev_ief_quantity = prev_qqq_or_shy_quantity = qqq_or_shy_quantity = 0
    prev_qqq_or_shy_price = remain_amount = 0
    prev_target_ticker = ''

    df_rebalancing_target = df_asset_laa[['IWD', 'GLD', 'IEF', my_etf_ticker, 'SHY', 'DATE']]
    output = pd.DataFrame()

    rebalancing_yearly_dates = [begin_date_of_investment] + rebalancing_yearly_dates
    iwd_quantity = gld_quantity = ief_quantity = 0
    iwd_amount = gld_amount = ief_amount = 0

    for date in rebalancing_monthly_dates:
        # @Tip: print(type(date), date)  # <class 'pandas._libs.tslibs.timestamps.Timestamp'> 2009-01-30 00:00:00
        df_price = df_rebalancing_target[df_rebalancing_target['DATE'] == date]

        spy_price_ma = rebalancing_monthly_spy_ma_200d[rebalancing_monthly_spy_ma_200d['DATE'] == date]
        # @Tip: .dt is needed when it's a group of data, if it's only one element you don't need .dt
        df_uem['YEAR'] = df_uem.loc[:, 'YEAR'].astype(str)
        df_uem['YEAR_MONTH'] = df_uem.loc[:, 'YEAR_MONTH'].astype(str)
        
        
        # 올바른 년도와 월 문자열 포맷 생성
        year_month_str = str(date.year) + str(date.month).zfill(2)
        uem_target = df_uem[(df_uem['YEAR'] == str(date.year)) & (df_uem['YEAR_MONTH'] == year_month_str)]


        if uem_target.shape[0] == 0:
            # print('UEM {} EMPTY'.format(date))
            continue

        iwd_price = df_price['IWD'].item()
        gld_price = df_price['GLD'].item()
        ief_price = df_price['IEF'].item()

        spy_index = spy_price_ma['SPY'].item()
        spy_ma_200d = spy_price_ma['MA_200D'].item()
        uem_index = uem_target['UNEMPLOYMENT_RATE'].item()
        uem_12m = uem_target['MA_12M'].item()
        
        # 사용자 지정 ETF를 포함한 타이밍 자산 선택
        target_ticker = 'my_etf_ticker'
        if spy_index < spy_ma_200d and uem_index > uem_12m:
            target_ticker = 'SHY'
        else:
            target_ticker = my_etf_ticker  # QQQ 대신 사용자 지정 ETF 사용
        qqq_or_shy_price = df_price[target_ticker].item()

        # IWD or GLD or IEF - Yearly ReBalancing
        if date in rebalancing_yearly_dates:
            if prev_qqq_or_shy_quantity == 0:
                allocation = budget / 4
            else:
                budget = ((prev_iwd_quantity * iwd_price) + (prev_gld_quantity * gld_price) + (
                        prev_ief_quantity * ief_price) + (prev_qqq_or_shy_quantity * qqq_or_shy_price)
                          + prev_cash_amount)
                allocation = budget / 4

            iwd_quantity = int(allocation / iwd_price)
            gld_quantity = int(allocation / gld_price)
            ief_quantity = int(allocation / ief_price)
            qqq_or_shy_quantity = int(allocation / qqq_or_shy_price)

            iwd_amount = iwd_price * iwd_quantity
            gld_amount = gld_price * gld_quantity
            ief_amount = ief_price * ief_quantity
            qqq_or_shy_amount = qqq_or_shy_price * qqq_or_shy_quantity

            prev_qqq_or_shy_quantity, prev_qqq_or_shy_price = qqq_or_shy_quantity, qqq_or_shy_price
            prev_iwd_quantity, prev_gld_quantity, prev_ief_quantity = iwd_quantity, gld_quantity, ief_quantity
            remain_amount = budget - sum([iwd_amount, gld_amount, ief_amount, qqq_or_shy_amount])
            prev_cash_amount = remain_amount
        else:
            # QQQ or SHY - Monthly ReBalancing
            iwd_amount = iwd_price * prev_iwd_quantity
            gld_amount = gld_price * prev_gld_quantity
            ief_amount = ief_price * prev_ief_quantity

            qqq_or_shy_price = df_price[target_ticker].item()
            if target_ticker == prev_target_ticker:
                qqq_or_shy_amount = prev_qqq_or_shy_quantity * qqq_or_shy_price
            else:
                qqq_or_shy_allocation = prev_qqq_or_shy_quantity * prev_qqq_or_shy_price
                qqq_or_shy_quantity = int(qqq_or_shy_allocation / qqq_or_shy_price)
                qqq_or_shy_amount = qqq_or_shy_quantity * qqq_or_shy_price  # qqq_or_shy_allocation

                prev_qqq_or_shy_quantity, prev_qqq_or_shy_price = qqq_or_shy_quantity, qqq_or_shy_price

        prev_target_ticker = target_ticker

        # Store Result
        price_dict = {'IWD_P': iwd_price, 'GLD_P': gld_price, 'IEF_P': ief_price,
                      'QQQ_SHY_P': qqq_or_shy_price}
        quantity_dict = {'IWD_Q': iwd_quantity, 'GLD_Q': gld_quantity, 'IEF_Q': ief_quantity,
                         'QQQ_SHY_Q': qqq_or_shy_quantity}
        amount_dict = {'IWD': iwd_amount, 'GLD': gld_amount, 'IEF': ief_amount, 'QQQ_SHY': qqq_or_shy_amount,
                       'CASH': remain_amount}

        # Your dictionary to append as a new row
        output_dict = {'DATE': date, **price_dict, **quantity_dict, **amount_dict,
                    'BUDGET': sum([iwd_amount, gld_amount, ief_amount, qqq_or_shy_amount, remain_amount]),
                    'TARGET': target_ticker,
                    'TOTAL': sum([iwd_amount, gld_amount, ief_amount, qqq_or_shy_amount])}

        # Append the new row to the DataFrame
        output = pd.concat([output, pd.DataFrame([output_dict])], ignore_index=True)

    return output

# Assuming df_asset is your DataFrame and it's already prepared
def graph_cagr_mdd(df_asset, file_name='output'):
    trade_month = round(len(df_asset.index) / 12, 2)

    if df_asset['TOTAL'].iat[0] != float(0):
        total_profit = (df_asset['TOTAL'].iat[-1] / df_asset['TOTAL'].iat[0])
        cagr = round((total_profit ** (1 / trade_month) - 1) * 100, 2)

        df_asset['DRAWDOWN'] = (-(df_asset['TOTAL'].cummax() - df_asset['TOTAL']) / df_asset['TOTAL'].cummax()) * 100
        df_asset['YEAR'] = df_asset['DATE'].dt.strftime('%Y-%m').astype(str)

        print('MM_PERIOD', trade_month, 'CAGR', cagr, 'MDD', df_asset['DRAWDOWN'].min())

        # Creating the subplot figure with 2 rows
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('CAGR and MDD Analysis', fontsize=16)

        # Plotting Total Returns
        axs[0].set_title('Total Returns')
        sns.lineplot(data=df_asset, x='DATE', y='TOTAL', ax=axs[0], color='blue', linewidth=2.5)
        axs[0].set_ylabel('Total Return', fontsize=12)
        axs[0].grid(True)

        # Plotting Drawdown
        axs[1].set_title('Maximum Drawdown')
        sns.lineplot(data=df_asset, x='DATE', y='DRAWDOWN', ax=axs[1], color='red', linewidth=2.5)
        axs[1].set_ylabel('Drawdown %', fontsize=12)
        axs[1].grid(True)

        # Formatting the date on x-axis
        date_format = DateFormatter("%Y-%m")
        axs[0].xaxis.set_major_formatter(date_format)
        axs[1].xaxis.set_major_formatter(date_format)

        # Rotating date labels for better readability
        plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)
        plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Saving the plot
        plt.savefig(file_name + '.png', dpi=600)
        # plt.show()

def graph_cagr_mdd_single_ax(df_asset, file_name='output'):
    # Calculate trading months and other metrics if necessary
    trade_month = round(len(df_asset.index) / 12, 2)
    # Additional calculations like CAGR and MDD can be added here
    
    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Formatting Date on x-axis
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax1.set_xlabel('Date')
    
    # Plot Total Returns on ax1
    color = 'tab:blue'
    ax1.set_ylabel('Total Return', color=color)
    sns.lineplot(data=df_asset, x='DATE', y='TOTAL', ax=ax1, color=color, linewidth=2.5, label='Total Return')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a second y-axis for MDD
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Drawdown (%)', color=color)
    sns.lineplot(data=df_asset, x='DATE', y='DRAWDOWN', ax=ax2, color=color, linewidth=2.5, label='MDD')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Additional plot formatting
    fig.tight_layout()
    plt.title('Total Return and Maximum Drawdown')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    
    # Save the plot
    plt.savefig(f"{file_name}.png", dpi=600)
    # plt.show()

# Ensure your df_asset DataFrame is correctly prepared with 'DATE', 'TOTAL', and 'DRAWDOWN' columns
# graph_cagr_mdd_single_ax(df_asset, 'output_filename')

if __name__ == '__main__':
    # @Tip: BIL beginning date 2007/05/30
    #       GLD beginning date 2004/11/18
    start = datetime(2007, 5, 30)
    end = datetime(datetime.now().year, datetime.now().month, datetime.now().day)
    df_asset_laa, df_asset_dual, uem_monthly = asset_data(start, end)

    print('LAA -- START')
    
    # 수동으로 읽기
    df_asset_laa = pd.read_csv('df_asset_laa.csv')
    df_asset_laa['DATE'] = pd.to_datetime(df_asset_laa['DATE'])
    
    my_etf_name = 'PriceTrends_5'
    
    df_output = laa_backtest(df_asset_laa, uem_monthly, my_etf_name)
    df_output = df_output[df_output['TOTAL'] != 0]
    
    graph_cagr_mdd(df_output, f'{my_etf_name}_output_1')
    graph_cagr_mdd_single_ax(df_output, f'{my_etf_name}_output_2')
    
    filename = 'laa-output-' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx'
    df_output.to_excel('.' + os.sep + filename, sheet_name='rtn')
    print('LAA -- END')

    # print('UEM QQQ -- START')
    # df_output = laa_variant_qqq_backtest(df_asset_laa, uem_monthly)
    # graph_cagr_mdd(df_output, 'UEM_QQQ_output.png')
    # filename = 'laa-variant-output-' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx'
    # df_output.to_excel('.' + os.sep + filename, sheet_name='rtn')
    # print('UEM QQQ -- END')

    # print('DUAL MOMENTUM -- START')
    # df_output = dual_momentum_backtest(df_asset_dual)
    # filename = 'dual-mtum-output-' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx'
    # df_output.to_excel('.' + os.sep + filename, sheet_name='rtn')
    # graph_cagr_mdd(df_output, 'DUAL_MTM_output.png')
    # print('DUAL MOMENTUM -- END')

    # print('UEM SMH -- START')
    # df_output = laa_variant_smh_backtest(df_asset_laa, uem_monthly)
    # graph_cagr_mdd(df_output, 'UEM_SMH_output.png')
    # filename = 'smh-variant-output-' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx'
    # df_output.to_excel('.' + os.sep + filename, sheet_name='rtn')
    # print('UEM SMH -- END')