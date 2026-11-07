import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

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
# 2. LOGIC TÍNH TOÁN CƠ BẢN
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

def detect_touches(df, ma_series, mode='intraday'):
    touches = []
    valid_indices = df.index[ma_series.notna()]
    if len(valid_indices) < 2: return touches

    for i in range(1, len(df)):
        current_date = df.index[i]
        ma = ma_series.iloc[i]
        if pd.isna(ma): continue

        low, high, close = df['low'].iloc[i], df['high'].iloc[i], df['close'].iloc[i]
        prev_close, prev_ma = df['close'].iloc[i - 1], ma_series.iloc[i - 1]

        crossing = False
        if not pd.isna(prev_ma):
            crossing = ((prev_close < prev_ma and close > ma) or (prev_close > prev_ma and close < ma))

        if mode == 'intraday':
            is_touch = (low <= ma <= high) or crossing
        else:
            is_touch = (close == ma) or crossing

        if is_touch:
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
# 3. WALK-FORWARD BACKTEST LOGIC
# ==========================================
def run_walk_forward_backtest(all_data, target_ind='SMA1250', mode='intraday'):
    """
    Chạy mô phỏng Walk-Forward Analysis lùi lại 1Y, 2Y, 3Y, 4Y.
    Tạo snapshot quá khứ -> Lập danh mục Kelly -> Chạy thực tế -> So sánh Top 3 Hindsight.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    timeframes = {'1Y': 1, '2Y': 2, '3Y': 3, '4Y': 4}
    results = {}
    
    for tf_label, years_back in timeframes.items():
        # Mốc cắt dữ liệu (T_0)
        snapshot_date = now - relativedelta(years=years_back)
        
        past_metrics_by_stock = {}
        forward_performance = {}
        
        for ticker, df in all_data.items():
            if df.empty: continue
            
            # 1. CẮT DỮ LIỆU TỚI snapshot_date
            past_df = df[df.index <= snapshot_date]
            if len(past_df) < 50: continue # Quá ít dữ liệu
            
            # Tính SMA trên quá khứ
            ma_series_past = compute_ma(past_df, target_ind)
            if ma_series_past.isna().all(): continue
            
            # Tìm điểm chạm trong quá khứ để xây dựng Kelly
            touches = detect_touches(past_df, ma_series_past, mode=mode)
            touch_metrics = [m for m in (compute_touch_metrics(past_df, t['idx'], t['price_at_touch']) for t in touches) if m]
            if not touch_metrics: continue
            
            stats = compute_stats(touch_metrics)
            if stats:
                past_metrics_by_stock[ticker] = stats
                
            # 2. LẤY DIỄN BIẾN TƯƠNG LAI (TỪ snapshot_date ĐẾN HIỆN TẠI)
            future_df = df[df.index >= snapshot_date]
            if len(future_df) > 0:
                buy_price = future_df['close'].iloc[0]
                current_price = future_df['close'].iloc[-1]
                min_price = future_df['low'].min()
                
                perf = (current_price / buy_price - 1) * 100
                max_dd = (min_price / buy_price - 1) * 100
                
                forward_performance[ticker] = {
                    'perf': round(perf, 1),
                    'max_dd': round(max_dd, 1)
                }

        # 3. MÔ PHỎNG DANH MỤC KELLY
        kelly_candidates = []
        for ticker, stats in past_metrics_by_stock.items():
            if stats['half_kelly'] > 0 and ticker in forward_performance:
                kelly_candidates.append({
                    'ticker': ticker,
                    'half_kelly': stats['half_kelly'],
                    'perf': forward_performance[ticker]['perf'],
                    'max_dd': forward_performance[ticker]['max_dd']
                })
                
        total_kelly_score = sum(c['half_kelly'] for c in kelly_candidates)
        portfolio_perf = 0
        portfolio_dd_approx = 0
        allocated_pct = 0
        
        if kelly_candidates:
            # Sắp xếp Kelly từ cao xuống thấp
            kelly_candidates.sort(key=lambda x: x['half_kelly'], reverse=True)
            
            # Chuẩn hóa tỷ trọng (giới hạn max 100%)
            for c in kelly_candidates:
                c['weight_pct'] = (c['half_kelly'] / max(1, total_kelly_score)) * min(100, total_kelly_score)
                portfolio_perf += (c['weight_pct'] / 100) * c['perf']
                # Ước lượng Max DD danh mục (Bình quân gia quyền DD các mã - tính tương đối)
                portfolio_dd_approx += (c['weight_pct'] / 100) * c['max_dd']
                
            allocated_pct = min(100, total_kelly_score)
            
        # 4. TÌM TOP 3 HINDSIGHT (THỰC TẾ XẢY RA)
        hindsight_ranking = []
        for ticker, fw_perf in forward_performance.items():
             hindsight_ranking.append({
                 'ticker': ticker,
                 'perf': fw_perf['perf'],
                 'max_dd': fw_perf['max_dd']
             })
        hindsight_ranking.sort(key=lambda x: x['perf'], reverse=True)
        top_3_hindsight = hindsight_ranking[:3]
        
        results[tf_label] = {
            'snapshot_date': snapshot_date.strftime('%Y-%m-%d'),
            'kelly_portfolio': {
                'total_perf': round(portfolio_perf, 1),
                'approx_max_dd': round(portfolio_dd_approx, 1),
                'allocated_pct': round(allocated_pct, 1),
                'holdings': kelly_candidates
            },
            'hindsight_top_3': top_3_hindsight
        }
        
    return results

# ==========================================
# 4. HÀM CHÍNH
# ==========================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for stock in STOCKS_LIST:
        try:
            df = yf.download(f"{stock}.VN", start="2000-01-01", auto_adjust=True, progress=False, multi_level_index=False)
            if not df.empty: df.to_csv(f"{DATA_DIR}/{stock}.csv")
        except Exception as e: print(f"Lỗi tải {stock}: {e}")

    all_data = {}
    for file_name in [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]:
        try:
            stock_symbol = file_name.replace('.csv', '')
            if stock_symbol not in STOCKS_LIST: continue
            
            df = pd.read_csv(f"{DATA_DIR}/{file_name}", header=0)
            if 'Ticker' in df.columns or df.iloc[0].astype(str).str.contains('VN').any():
                 df = df.iloc[1:].reset_index(drop=True)
            
            if len(df) < 10: continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            df['time'] = pd.to_datetime(df[df.columns[0]], errors='coerce')
            df.set_index('time', inplace=True)
            df = df.sort_index().dropna(subset=['close'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            all_data[stock_symbol] = df[df['volume'] > 0]
        except Exception as e: pass

    result = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tickers_count": len(all_data), "indicators": {}, "walk_forward": {}}

    for ind_name in INDICATORS_CONFIG.keys():
        ind_stocks_close = []
        ind_stocks_intraday = []
        
        for ticker, df in all_data.items():
            ma_series = compute_ma(df, ind_name)
            if ma_series.isna().all(): continue
            
            # --- Chạm Trong phiên ---
            touches_intra = detect_touches(df, ma_series, mode='intraday')
            metrics_intra = [m for m in (compute_touch_metrics(df, t['idx'], t['price_at_touch']) for t in touches_intra) if m]
            if metrics_intra:
                metrics_intra.sort(key=lambda x: x['date'], reverse=True)
                ind_stocks_intraday.append({
                    "ticker": ticker, "group": get_stock_group(ticker),
                    **compute_current(df, ma_series), "stats": compute_stats(metrics_intra), "touches": metrics_intra
                })
                
            # --- Chạm Đóng cửa ---
            touches_close = detect_touches(df, ma_series, mode='close')
            metrics_close = [m for m in (compute_touch_metrics(df, t['idx'], t['price_at_touch']) for t in touches_close) if m]
            if metrics_close:
                metrics_close.sort(key=lambda x: x['date'], reverse=True)
                ind_stocks_close.append({
                    "ticker": ticker, "group": get_stock_group(ticker),
                    **compute_current(df, ma_series), "stats": compute_stats(metrics_close), "touches": metrics_close
                })
                
        result["indicators"][ind_name] = {
            "intraday": ind_stocks_intraday,
            "close": ind_stocks_close
        }

    # Bổ sung chạy Walk-Forward Backtest cho SMA1250 (Intraday) làm tiêu chuẩn
    result["walk_forward"] = run_walk_forward_backtest(all_data, target_ind='SMA1250', mode='intraday')

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
