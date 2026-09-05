"""Regras transparentes de triagem; não representam um modelo validado de retorno."""
import numpy as np
import pandas as pd


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def prices(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if not {'Close', 'High', 'Low', 'Volume'}.issubset(frame.columns):
        return pd.DataFrame()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    for col in ['Close', 'High', 'Low', 'Volume']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return frame.dropna(subset=['Close', 'High', 'Low']).loc[lambda x: x.Close > 0]


def technical(frame, benchmark, today=None):
    df, market = prices(frame), prices(benchmark)
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    if len(df) < 220:
        return {'available': False, 'reason': 'Precisamos de pelo menos 220 pregões para avaliar a tendência maior.'}
    age = (today - df.index[-1].normalize()).days
    if age > 7 or age < 0:
        return {'available': False, 'reason': 'Os preços estão desatualizados ou têm uma data inconsistente.'}
    close = df.Close
    ma20, ma50, ma200 = [close.rolling(n).mean() for n in (20, 50, 200)]
    current = float(close.iloc[-1])
    long_up = ma200.iloc[-1] > ma200.iloc[-21]
    short_up = ma50.iloc[-1] > ma50.iloc[-21]
    up = current > ma50.iloc[-1] > ma200.iloc[-1] and long_up and short_up
    down = current < ma50.iloc[-1] < ma200.iloc[-1] and not long_up
    stage = 'Alta confirmada pelos filtros' if up else 'Queda predominante' if down else 'Recuperação ainda incompleta' if current > ma20.iloc[-1] else 'Sem direção clara'
    # Compare volume only to preceding sessions. A sell-off never earns confirmation.
    volume_base = df.Volume.shift(1).rolling(20).mean().iloc[-1]
    volume_ratio = number(df.Volume.iloc[-1] / volume_base) if volume_base > 0 else None
    buying_volume = bool(close.iloc[-1] > close.iloc[-2] and volume_ratio is not None and volume_ratio >= 1.2)
    support = float(df.Low.iloc[-21:-1].min())
    resistance = float(df.High.iloc[-61:-1].max())
    # Both distances have the purchase price as denominator.
    loss_to_support = (current - support) / current * 100
    room = (resistance - current) / current * 100
    rr = room / loss_to_support if loss_to_support > 0 and room > 0 else None
    returns = close.pct_change().dropna().tail(60)
    volatility = float(returns.std() * np.sqrt(252) * 100)
    turnover = number((df.Close * df.Volume).tail(20).mean())
    risk_high = volatility > 45 or turnover is None or turnover < 2_000_000 or loss_to_support > 12 or loss_to_support <= 0
    year = df.loc[df.index >= df.index[-1] - pd.Timedelta(days=365)]
    high, low = float(year.High.max()), float(year.Low.min())
    market_ok, relative = None, None
    if len(market) >= 220 and 0 <= (today - market.index[-1].normalize()).days <= 7:
        bm = market.Close
        bm200 = bm.rolling(200).mean()
        market_ok = bool(bm.iloc[-1] > bm200.iloc[-1] and bm200.iloc[-1] > bm200.iloc[-21])
        aligned = pd.concat([close.rename('stock'), bm.rename('market')], axis=1).dropna()
        if len(aligned) >= 64:
            relative = float(((aligned.stock.iloc[-1] / aligned.stock.iloc[-64]) - (aligned.market.iloc[-1] / aligned.market.iloc[-64])) * 100)
    return dict(available=True, stage=stage, up=bool(up), down=bool(down), price=current,
                date=df.index[-1], ma20=float(ma20.iloc[-1]), ma50=float(ma50.iloc[-1]), ma200=float(ma200.iloc[-1]),
                buying_volume=buying_volume, volume_ratio=volume_ratio, support=support, resistance=resistance,
                loss=loss_to_support, room=room, rr=rr, volatility=volatility, turnover=turnover,
                risk_high=bool(risk_high), low=low, high=high, market_ok=market_ok, relative=relative,
                chart=pd.DataFrame({'Preço': close, 'Média de 50 pregões': ma50, 'Média de 200 pregões': ma200}).tail(252))


def row_values(frame, name):
    if frame is None or frame.empty or name not in frame.index:
        return []
    series = frame.loc[name].dropna().sort_index(ascending=False)
    return [(pd.Timestamp(d).tz_localize(None), number(v)) for d, v in series.items() if number(v) is not None]


def fundamentals(info, income, balance, cash, today=None):
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    profits = row_values(income, 'NetIncome')
    revenues = row_values(income, 'TotalRevenue')
    flows = row_values(cash, 'OperatingCashFlow')
    debts = row_values(balance, 'TotalDebt')
    equity = row_values(balance, 'StockholdersEquity')
    dates = [v[0][0] for v in [profits, revenues, flows, debts, equity] if v]
    financial = info.get('sector') == 'Financial Services'
    reasons, gaps = [], []
    if financial:
        gaps.append('Bancos e seguradoras exigem indicadores próprios de capital, crédito e solvência, ainda não integrados.')
    if len(profits) < 2 or len(revenues) < 2 or not flows or not debts or not equity:
        gaps.append('Faltam demonstrações anuais suficientes para avaliar lucros, caixa e dívida.')
    if dates and any((today - d).days > 550 or (today - d).days < 0 for d in dates):
        gaps.append('Há demonstrações antigas ou datas inconsistentes.')
    if dates and (max(dates) - min(dates)).days > 120:
        gaps.append('As demonstrações disponíveis cobrem períodos diferentes.')
    if not info.get('sector'):
        gaps.append('Setor da empresa não identificado.')
    if info.get('currency') != 'BRL' or info.get('financialCurrency') != 'BRL':
        gaps.append('Moedas dos preços e balanços não confirmadas em reais.')
    profit = profits[0][1] if profits else None
    flow = flows[0][1] if flows else None
    debt_equity = debts[0][1] / equity[0][1] if debts and equity and equity[0][1] > 0 else None
    growth = (revenues[0][1] / revenues[1][1] - 1) * 100 if len(revenues) > 1 and revenues[1][1] > 0 else None
    if not gaps:
        if profit <= 0:
            reasons.append('A empresa registrou prejuízo no último exercício disponível.')
        if flow <= 0:
            reasons.append('A operação consumiu caixa no último exercício disponível.')
        if equity[0][1] <= 0:
            reasons.append('O patrimônio líquido informado é nulo ou negativo.')
        if debt_equity is not None and debt_equity > 2:
            reasons.append('A dívida total supera duas vezes o patrimônio: exige avaliação específica do setor.')
        if growth is not None and growth < -10:
            reasons.append('A receita anual caiu mais de 10%.')
    recurring = len(profits) >= 3 and all(v > 0 for _, v in profits[:3])
    state = 'Análise incompleta' if gaps else 'Exige atenção' if reasons else 'Indicadores básicos positivos' if recurring else 'Histórico de lucros ainda irregular'
    return dict(state=state, gaps=gaps, reasons=reasons, profit=profit, flow=flow,
                debt_equity=debt_equity, growth=growth, date=max(dates) if dates else None,
                recurring=recurring, financial=financial)


def conclusion(tech, fund, valuation):
    if not tech.get('available'):
        return '⚪ DADOS INSUFICIENTES', tech['reason']
    if tech['down']:
        return '🔴 DESFAVORÁVEL À COMPRA AGORA', 'A queda ainda predomina. Estar perto da mínima não confirma uma oportunidade.'
    if fund['reasons']:
        return '🔴 DESFAVORÁVEL À COMPRA AGORA', fund['reasons'][0]
    if fund['gaps'] or tech['market_ok'] is None:
        return '⚪ ANÁLISE INCOMPLETA', 'Ainda faltam dados essenciais. Os sinais de preço abaixo não bastam para concluir a compra.'
    if tech['risk_high']:
        return '🟡 AGUARDAR — RISCO ELEVADO', 'A oscilação, a liquidez ou a distância até o fundo recente exigem cautela.'
    if not tech['up'] or not tech['market_ok'] or tech['relative'] is None or tech['relative'] <= 0:
        return '🟡 AGUARDAR CONFIRMAÇÃO', 'A tendência da ação e o contexto do mercado ainda não estão alinhados.'
    return '🟡 SINAIS POSITIVOS — COMPRA NÃO CONFIRMADA', 'Há sinais favoráveis, mas a avaliação de preço, os fatos relevantes e a validação do método ainda não permitem concluir pela compra.'
