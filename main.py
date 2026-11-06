import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_DIR = "./data"
OUTPUT_JSON = "./backtest_results.json"

STOCKS_LIST = [
    # 1. Ngân hàng
    "ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "NAB", 
    "OCB", "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB", "BAB",
    # 2. Khác
    "GAS", "HCM", "EVF", "SSI", "VCI", "VIX", "VND"
]

STOCK_GROUPS = {
    "Ngân hàng": ["ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "NAB", "OCB", "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB", "BAB"],
    "Khác": ["GAS", "HCM", "EVF", "SSI", "VCI", "VIX", "VND"]
}


INDICATORS_CONFIG = {
    'SMA1250': {'type': 'sma', 'period': 1250},
    'SMA1000': {'type': 'sma', 'period': 1000},
    'SMA400': {'type': 'sma', 'period': 400},
    'EMA400': {'type': 'ema', 'period': 400},
    'SMA250': {'type': 'sma', 'period': 250},
    'SMA200': {'type': 'sma', 'period': 200},
}

def get_stock_group(ticker):
    for group, tickers in STOCK_GROUPS.items():
        if ticker in tickers: return group
    return "Khác"

# ==========================================
# 2. LOGIC TÍNH TOÁN
# ==========================================
def compute_ma(df, ind_name):
    if df.empty or 'close' not in df.columns: return pd.Series(dtype='float64')
    if ind_name.startswith('SMA'):
        period = int(ind_name.replace('SMA', ''))
        return df['close'].rolling(window=period).mean()
    elif ind_name.startswith('EMA'):
        period = int(ind_name.replace('EMA', ''))
        return df['close'].ewm(span=period, adjust=False).mean()
    return pd.Series(dtype='float64')

def compute_current(df, ma_series):
    if df.empty or ma_series.empty or pd.isna(ma_series.iloc[-1]):
        return {'current_price': None, 'line_price': None, 'distance_pct': None, 'distance_abs_pct': None, 'direction': None}
    current_price = float(df['close'].iloc[-1])
    line_price = float(ma_series.iloc[-1])
    distance_pct = round((current_price / line_price - 1) * 100, 2)
    return {
        'current_price': current_price, 'line_price': line_price,
        'distance_pct': distance_pct, 'distance_abs_pct': abs(distance_pct),
        'direction': 'above' if distance_pct > 0 else 'below'
    }

def detect_touches(df, ma_series):
    touches = []
    valid_indices = df.index[ma_series.notna()]
    if len(valid_indices) < 2: return touches
        
    for i in range(1, len(df)):
        current_date = df.index[i]
        ma = ma_series.iloc[i]
        if pd.isna(ma): continue
            
        low, high, close = df['low'].iloc[i], df['high'].iloc[i], df['close'].iloc[i]
        prev_close, prev_ma = df['close'].iloc[i - 1], ma_series.iloc[i - 1]
        
        intraday_touch = low <= ma <= high
        crossing = False
        if not pd.isna(prev_ma):
            crossing = ((prev_close < prev_ma and close > ma) or (prev_close > prev_ma and close < ma))
            
        if intraday_touch or crossing:
            if not touches or (current_date - touches[-1]['date_obj']).days > 5:
                touches.append({'date_obj': current_date, 'idx': i, 'price_at_touch': close})
    return touches

def compute_touch_metrics(df, touch_idx, price_at_touch, max_horizon=250):
    end_idx = min(touch_idx + max_horizon, len(df) - 1)
    forward_close = df['close'].iloc[touch_idx + 1 : end_idx + 1]
    forward_high = df['high'].iloc[touch_idx + 1 : end_idx + 1]
    forward_low = df['low'].iloc[touch_idx + 1 : end_idx + 1]
    
    if len(forward_close) == 0: return None
        
    def get_perf(n):
        if len(forward_close) >= n: return round((forward_close.iloc[n-1] / price_at_touch - 1) * 100, 1)
        return None
    
    highs_pct = [(h / price_at_touch - 1) * 100 for h in forward_high]
    lows_pct = [(l / price_at_touch - 1) * 100 for l in forward_low]
    
    max_gain = round(max(highs_pct), 1) if highs_pct else 0
    max_gain_day = highs_pct.index(max(highs_pct)) + 1 if highs_pct else 0
    max_dd = round(min(lows_pct), 1) if lows_pct else 0
    max_dd_day = lows_pct.index(min(lows_pct)) + 1 if lows_pct else 0
    
    current_perf = round((forward_close.iloc[-1] / price_at_touch - 1) * 100, 1)
    duration = len(forward_close)
    touch_date = df.index[touch_idx]
    
    return {
        'date': touch_date.strftime('%Y-%m-%d'),
        'days_ago': (datetime.now() - touch_date).days,
        'price_at_touch': float(price_at_touch),
        'perf_30d': get_perf(30), 'perf_60d': get_perf(60),
        'perf_120d': get_perf(120), 'perf_250d': get_perf(250),
        'max_gain': float(max_gain), 'max_gain_day': max_gain_day,
        'max_dd': float(max_dd), 'max_dd_day': max_dd_day,
        'current_perf': float(current_perf),
        'duration': duration, 'ongoing': duration < max_horizon
    }

def compute_stats(touches_metrics):
    if not touches_metrics: return None
    closed = [t for t in touches_metrics if t.get('perf_60d') is not None]
    wins = [t for t in closed if t['perf_60d'] > 0]
    losses = [t for t in closed if t['perf_60d'] <= 0]
    
    win_rate = round(len(wins) / len(closed) * 100, 1) if closed else 0
    avg_win = round(sum(t['perf_60d'] for t in wins) / len(wins), 1) if wins else 0
    avg_loss = round(sum(t['perf_60d'] for t in losses) / len(losses), 1) if losses else 0
    rr = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0
    
    kelly = 0
    if rr > 0:
        win_prob = win_rate / 100
        kelly = max(0, win_prob - ((1 - win_prob) / rr))
        
    best = max(touches_metrics, key=lambda t: t['max_gain'])
    worst = min(touches_metrics, key=lambda t: t['max_dd'])
    last = touches_metrics[0] if touches_metrics else None
    
    return {
        'count': len(touches_metrics), 'win_rate': float(win_rate),
        'avg_win': float(avg_win), 'avg_loss': float(avg_loss), 'rr': float(rr),
        'kelly': float(round(kelly * 100, 1)), 'half_kelly': float(round((kelly * 100) / 2, 1)),
        'avg_max_gain': float(round(sum(t['max_gain'] for t in touches_metrics) / len(touches_metrics), 1)),
        'avg_max_dd': float(round(sum(t['max_dd'] for t in touches_metrics) / len(touches_metrics), 1)),
        'avg_gain_day': int(sum(t['max_gain_day'] for t in touches_metrics) / len(touches_metrics)),
        'avg_dd_day': int(sum(t['max_dd_day'] for t in touches_metrics) / len(touches_metrics)),
        'days_since_last': last['days_ago'] if last else None,
        'last_touch_date': last['date'] if last else None,
        'best_touch': {'max_gain': float(best['max_gain']), 'max_gain_day': int(best['max_gain_day']), 'date': best['date']},
        'worst_touch': {'max_dd': float(worst['max_dd']), 'max_dd_day': int(worst['max_dd_day']), 'date': worst['date']}
    }

# ==========================================
# 3. LUỒNG CHẠY CHÍNH
# ==========================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for stock in STOCKS_LIST:
        try:
            df = yf.download(f"{stock}.VN", start="2000-01-01", auto_adjust=True, progress=False)
            if not df.empty: df.to_csv(f"{DATA_DIR}/{stock}.csv")
        except Exception as e: print(f"Lỗi tải {stock}: {e}")

    all_data = {}
    for file_name in [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]:
        try:
            stock_symbol = file_name.replace('.csv', '')
            
            # Bỏ qua các file CSV của mã cổ phiếu không có trong danh sách hiện tại
            if stock_symbol not in STOCKS_LIST:
                continue
                
            df = pd.read_csv(f"{DATA_DIR}/{file_name}")
            if len(df) < 10: continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            df['time'] = pd.to_datetime(df[df.columns[0]], errors='coerce')
            df.set_index('time', inplace=True)
            df = df.sort_index().dropna(subset=['close'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            all_data[stock_symbol] = df[df['volume'] > 0]
        except Exception as e: pass
            
    result = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tickers_count": len(all_data), "indicators": {}}
    
    for ind_name in INDICATORS_CONFIG.keys():
        ind_stocks = []
        for ticker, df in all_data.items():
            ma_series = compute_ma(df, ind_name)
            if ma_series.isna().all(): continue
            touches = detect_touches(df, ma_series)
            touch_metrics = [m for m in (compute_touch_metrics(df, t['idx'], t['price_at_touch']) for t in touches) if m]
            if not touch_metrics: continue
            touch_metrics.sort(key=lambda x: x['date'], reverse=True)
            ind_stocks.append({
                "ticker": ticker, "group": get_stock_group(ticker),
                **compute_current(df, ma_series), "stats": compute_stats(touch_metrics), "touches": touch_metrics
            })
        result["indicators"][ind_name] = {"stocks": ind_stocks}
        
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
