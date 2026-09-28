"""Reproduce the preregistered Yahoo daily-data experiment (not investment advice)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from pairs_trading.backtest.engine import run_backtest
from pairs_trading.config import DataConfig, EvaluationConfig, ExecutionConfig, ExperimentConfig, ScreeningConfig, SignalConfig
from pairs_trading.data.validation import validate_market_data
from pairs_trading.research.cointegration import analyze_pair
from pairs_trading.research.screening import screen_pairs
from pairs_trading.reporting.metrics import calculate_metrics
from pairs_trading.reporting.outputs import export_results

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'frozen_protocol.yaml'


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_prices(p: dict, output: Path) -> tuple[pd.DataFrame, dict]:
    universe = p['universe']
    raw = yf.download(universe, start=p['start_inclusive'], end=p['end_exclusive'],
                      interval='1d', auto_adjust=True, actions=False, repair=False,
                      group_by='ticker', threads=False, progress=False, keepna=False)
    rows = []
    audit = {'source': p['source'], 'source_docs': 'https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html',
             'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
             'protocol_sha256': checksum(PROTOCOL), 'universe': universe,
             'auto_adjust': True, 'adjusted_open_field': 'Open', 'adjusted_close_field': 'Close',
             'start_inclusive': p['start_inclusive'], 'end_exclusive': p['end_exclusive']}
    for symbol in universe:
        frame = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) and symbol in raw.columns.get_level_values(0) else None
        if frame is None or frame.empty or not {'Open','Close'}.issubset(frame.columns):
            raise ValueError(f'Missing complete adjusted Open and Close for {symbol}; cannot replace universe after preregistration.')
        frame = frame[['Open','Close']].dropna()
        if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() <= 0).any():
            raise ValueError(f'Invalid adjusted OHLC for {symbol}')
        for date, row in frame.iterrows():
            rows.append({'timestamp': date.date().isoformat(), 'symbol': symbol,
                         'adjusted_open': float(row['Open']), 'adjusted_close': float(row['Close'])})
    data = pd.DataFrame(rows).sort_values(['timestamp', 'symbol'])
    dates = data.groupby('timestamp').symbol.nunique()
    if (dates != len(universe)).any():
        raise ValueError('Incomplete universe rows: no forward-fill or dropping individual symbols; inspect source.')
    audit['rows'] = len(data)
    audit['first_session'] = str(data.timestamp.min())
    audit['last_session'] = str(data.timestamp.max())
    audit['sessions'] = len(dates)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / 'adjusted_prices.csv', index=False, float_format='%.12g')
    audit['data_sha256'] = checksum(output / 'adjusted_prices.csv')
    (output / 'data_manifest.json').write_text(json.dumps(audit, indent=2)+'\n')
    return data, audit


def benchmark(market, start, end):
    prices = market.adjusted_close.loc[start:end]
    return float((prices.iloc[-1] / prices.iloc[0]).mean() - 1)


def experiment(market, p, output, label, train_end, start, end, cost_multiple=1, borrow=None):
    output.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        universe=tuple(p['universe']),
        data=DataConfig(min_observations=p['min_observations'], path='adjusted_prices.csv'),
        screening=ScreeningConfig(formation_window_days=p['formation_window_days'],
            engle_granger_significance=p['engle_granger_significance'],
            adf_significance=p['adf_significance'],
            refit_frequency_days=p['refit_frequency_days']),
        signals=SignalConfig(zscore_window_days=p['zscore_window_days'],
            entry_threshold=p['entry_threshold'], exit_threshold=p['exit_threshold'],
            stop_threshold=p['stop_threshold'], max_holding_days=p['max_holding_days']),
        execution=ExecutionConfig(commission_bps_per_side=p['commission_bps_per_side']*cost_multiple,
            slippage_bps_per_side=p['slippage_bps_per_side']*cost_multiple,
            annual_borrow_rate=p['annual_borrow_rate'] if borrow is None else borrow,
            initial_capital=p['initial_capital']),
        evaluation=EvaluationConfig(train_end=train_end, test_start=start, test_end=end))
    training = market.adjusted_close.loc[:train_end]
    candidates = screen_pairs(training, symbols=p['universe'],
        formation_window_days=p['formation_window_days'],
        min_observations=p['min_observations'],
        significance=p['engle_granger_significance'],
        adf_significance=p['adf_significance'])
    pd.DataFrame([asdict(c) for c in candidates]).to_csv(output/'pair_screening.csv', index=False)
    passing = sorted((c for c in candidates if c.cointegrated),
                     key=lambda c: (c.adjusted_pvalue, c.symbol_a, c.symbol_b))
    choice = passing[0] if passing else None
    evidence = {'label':label, 'train_end':train_end, 'evaluation_start':start,
        'evaluation_end':end, 'candidate_count':len(candidates), 'passing_count':len(passing),
        'rule':'alphabetically ordered pairs; Engle-Granger p <= 0.05 and BH adjusted p <= 0.05; choose lowest adjusted p, alphabetical tie-break; no pass => cash',
        'selected_pair':[choice.symbol_a,choice.symbol_b] if choice else None,
        'selected_adjusted_pvalue':choice.adjusted_pvalue if choice else None,
        'protocol_sha256':checksum(PROTOCOL), 'config':asdict(config)}
    (output/'selection.json').write_text(json.dumps(evidence, indent=2, default=str)+'\n')
    cash = 0.0
    buy_hold = benchmark(market,start,end)
    if choice is None:
        for name, columns in {'trades.csv':['entry_timestamp','exit_timestamp','side','net_pnl','total_costs','holding_bars','exit_reason','gross_pnl_before_costs'], 'fills.csv':['timestamp','symbol','quantity','reference_price','execution_price','commission','slippage_cost','reason']}.items():
            pd.DataFrame(columns=columns).to_csv(output/name,index=False)
        metrics = {'cumulative_net_return':cash, 'cumulative_gross_return':cash,
            'maximum_drawdown':cash, 'trade_count':0, 'status':'no pair passed; cash/no trade'}
    else:
        pair = analyze_pair(training.tail(p['formation_window_days']),
            choice.symbol_a, choice.symbol_b, min_observations=p['min_observations'],
            adf_significance=p['adf_significance'])
        result = run_backtest(market, pair, config.signals, config.execution,
            start_date=start, end_date=end,
            refit_frequency_days=p['refit_frequency_days'],
            formation_window_days=p['formation_window_days'],
            min_refit_observations=p['min_observations'])
        metrics = calculate_metrics(result.equity_curve,result.trades,result.fills,
            initial_capital=p['initial_capital'])
        export_results(result, metrics, output)
        ledger = pd.read_csv(output/'trades.csv')
        if not ledger.empty:
            ledger['gross_pnl_before_costs'] = ledger.net_pnl + ledger.total_costs
            ledger.to_csv(output/'trades.csv',index=False)
        if not np.isclose(metrics['ending_gross_equity']-metrics['ending_net_equity'],
                          metrics['total_commissions']+metrics['total_slippage_costs']+metrics['total_borrow_costs'], atol=.01):
            raise AssertionError('Gross/net cost reconciliation failed')
    metrics['equal_weight_universe_buy_hold_return'] = buy_hold
    metrics['cash_benchmark_return'] = cash
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    return {'label':label,'selected_pair':evidence['selected_pair'],
        'passing_count':len(passing), 'net_return':metrics['cumulative_net_return'],
        'gross_return':metrics['cumulative_gross_return'],
        'max_drawdown':metrics['maximum_drawdown'], 'trades':metrics['trade_count'],
        'equal_weight_buy_hold':buy_hold,'cash':cash,
        'cost_multiple':cost_multiple,'borrow_rate':config.execution.annual_borrow_rate}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/real_data'))
    parser.add_argument('--use-snapshot',action='store_true',help='Use previously saved adjusted_prices.csv without redownloading')
    args=parser.parse_args()
    p=yaml.safe_load(PROTOCOL.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    if args.use_snapshot:
        data=pd.read_csv(args.output/'adjusted_prices.csv')
        manifest=json.loads((args.output/'data_manifest.json').read_text())
        if checksum(args.output/'adjusted_prices.csv') != manifest['data_sha256']:
            raise ValueError('Snapshot hash mismatch')
        if manifest['protocol_sha256'] != checksum(PROTOCOL):
            raise ValueError('Protocol changed since snapshot')
    else:
        data,manifest=fetch_prices(p,args.output)
    market=validate_market_data(data,min_observations=p['min_observations'])
    result=[]
    result.append(experiment(market,p,args.output/'validation','validation',p['selection_training'][1],*p['validation']))
    for i, fold in enumerate(p['walk_forward'],1):
        result.append(experiment(market,p,args.output/f'walk_forward_{i}','walk_forward_'+str(i),
                     fold['train_end'],fold['evaluation_start'],fold['evaluation_end']))
    # Stress uses the same training-only selection and unchanged untouched-test dates.
    for multiple in p['stress']['transaction_cost_multipliers']:
        for borrow in p['stress']['annual_borrow_rates']:
            label=f'test_cost{multiple}x_borrow{int(borrow*100)}pct'
            result.append(experiment(market,p,args.output/label,label,
                        p['selection_training'][1],*p['untouched_test'],
                        cost_multiple=multiple,borrow=borrow))
    pd.DataFrame(result).to_csv(args.output/'summary.csv',index=False)
    print(pd.DataFrame(result).to_string(index=False))

if __name__=='__main__':main()
