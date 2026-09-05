
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
import re

st.set_page_config(page_title="Método Mirian - Monitor de Ações", page_icon="📈", layout="centered")

DEFAULT_TICKERS = ["SBSP3", "CLSC4", "VALE3", "VBBR3", "PETR4"]

def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker:
        return ticker
    return ticker if ticker.endswith(".SA") else ticker + ".SA"

@st.cache_data(ttl=900)
def load_data(ticker: str):
    yf_ticker = normalize_ticker(ticker)
    start = (date.today() - timedelta(days=730)).isoformat()
    df = yf.download(yf_ticker, start=start, auto_adjust=True, progress=False, threads=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.copy()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Price"])

    # Médias móveis
    df["MM20"] = df["Price"].rolling(20).mean()
    df["MM50"] = df["Price"].rolling(50).mean()
    df["MM200"] = df["Price"].rolling(200).mean()

    # RSI 14 (Wilder)
    delta = df["Price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = (100 - (100 / (1 + rs))).mask((avg_loss == 0) & (avg_gain > 0), 100).mask((avg_loss == 0) & (avg_gain == 0), 50)

    # MACD
    ema12 = df["Price"].ewm(span=12, adjust=False).mean()
    ema26 = df["Price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # Volume médio 20
    df["VOL20"] = df["Volume"].rolling(20).mean()

    return df

def pct(a, b):
    if b is None or b == 0 or pd.isna(b) or pd.isna(a):
        return np.nan
    return (a / b - 1) * 100

def fmt_brl(v):
    if pd.isna(v):
        return "—"
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(v):
    if pd.isna(v):
        return "—"
    return f"{v:.1f}%".replace(".", ",")

def price_region(pos):
    if pd.isna(pos):
        return "—"
    if pos <= 20:
        return "🟢 Muito baixa"
    if pos <= 35:
        return "🟢 Interessante"
    if pos <= 60:
        return "🟡 Intermediária"
    if pos <= 80:
        return "🟠 Elevada"
    return "🔴 Próxima da máxima"

def classify_total(score):
    if score >= 8.0:
        return "🟢🟢 Entrada muito interessante"
    if score >= 6.5:
        return "🟢 Entrada interessante"
    if score >= 4.5:
        return "🟡 Observar"
    return "🔴 Evitar / aguardar"

def analyze_ticker(ticker: str):
    df = load_data(ticker)
    if df.empty or len(df) < 30:
        return None, df

    last = df.iloc[-1]
    current = float(last["Price"])

    ytd = df[df.index.year == date.today().year]
    if ytd.empty:
        return None, df
    min_price = float(ytd["Low"].min()) if "Low" in ytd.columns else float(ytd["Price"].min())
    max_price = float(ytd["High"].max()) if "High" in ytd.columns else float(ytd["Price"].max())

    min_idx = ytd["Low"].idxmin() if "Low" in ytd.columns else ytd["Price"].idxmin()
    max_idx = ytd["High"].idxmax() if "High" in ytd.columns else ytd["Price"].idxmax()

    pos = ((current - min_price) / (max_price - min_price) * 100) if max_price > min_price else np.nan
    dist_min = (current / min_price - 1) * 100
    upside_max = (max_price / current - 1) * 100

    # Estrutura recente: regressão linear simples em 20 pregões
    recent = df.tail(20)
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent["Price"].values, 1)[0] if len(recent) >= 5 else 0

    mm20 = last["MM20"]
    mm50 = last["MM50"]
    mm200 = last["MM200"]
    rsi = last["RSI14"]
    macd = last["MACD"]
    signal = last["MACD_SIGNAL"]
    hist = last["MACD_HIST"]
    prev_hist = df["MACD_HIST"].iloc[-2] if len(df) >= 2 else np.nan
    vol20 = last["VOL20"]
    vol = last["Volume"]

    # Nota preço (0 a 5)
    price_score = 0.0
    if not pd.isna(pos):
        if pos <= 20: price_score += 2.5
        elif pos <= 35: price_score += 2.0
        elif pos <= 50: price_score += 1.3
        elif pos <= 70: price_score += 0.7
        else: price_score += 0.2

    if upside_max >= 25: price_score += 1.5
    elif upside_max >= 15: price_score += 1.0
    elif upside_max >= 8: price_score += 0.5

    rr = upside_max / dist_min if dist_min > 0 else np.nan
    if not pd.isna(rr):
        if rr >= 2: price_score += 1.0
        elif rr >= 1: price_score += 0.5

    price_score = min(5.0, price_score)

    # Nota tendência (0 a 5)
    trend_score = 0.0
    if not pd.isna(mm20) and current > mm20: trend_score += 0.8
    if not pd.isna(mm50) and current > mm50: trend_score += 0.8
    if not pd.isna(mm20) and not pd.isna(mm50) and mm20 > mm50: trend_score += 0.8
    if slope > 0: trend_score += 0.7

    if not pd.isna(rsi):
        if 50 <= rsi <= 65: trend_score += 0.9
        elif 45 <= rsi < 50 or 65 < rsi <= 72: trend_score += 0.5
        elif rsi > 72: trend_score += 0.2

    if not pd.isna(macd) and not pd.isna(signal):
        if macd > signal: trend_score += 0.7
        if macd > 0: trend_score += 0.4

    if not pd.isna(hist) and not pd.isna(prev_hist) and hist > prev_hist:
        trend_score += 0.4

    if not pd.isna(vol20) and vol20 > 0 and vol > vol20:
        trend_score += 0.3

    trend_score = min(5.0, trend_score)
    total = round(price_score + trend_score, 1)

    direction = "↗️ Para cima" if slope > 0 else ("↘️ Para baixo" if slope < 0 else "➡️ Lateral")

    result = {
        "Ticker": ticker.upper(),
        "Preço": current,
        "Mínima 2026": min_price,
        "Data mínima": min_idx.strftime("%d/%m/%Y"),
        "Máxima 2026": max_price,
        "Data máxima": max_idx.strftime("%d/%m/%Y"),
        "Posição na faixa": pos,
        "Distância da mínima": dist_min,
        "Potencial até máxima": upside_max,
        "MM20": mm20,
        "MM50": mm50,
        "MM200": mm200,
        "RSI14": rsi,
        "MACD": macd,
        "MACD Signal": signal,
        "MACD Hist": hist,
        "Volume atual": vol,
        "Volume médio 20": vol20,
        "Direção": direction,
        "Nota Preço": round(price_score, 1),
        "Nota Tendência": round(trend_score, 1),
        "Nota Total": total,
        "Classificação": classify_total(total),
        "Região": price_region(pos),
    }
    return result, df

st.title("📈 Método Mirian — Monitor de Ações")
st.caption("Procura ações em região de preço favorável que já estejam mostrando sinais de recuperação.")

with st.form("analisar"):
    ticker = st.text_input("Qual ação você quer analisar?", value="SBSP3", placeholder="Ex.: VALE3, CLSC4, PETR4").strip().upper()
    submitted = st.form_submit_button("ANALISAR AÇÃO", use_container_width=True, type="primary")

st.caption("Dados do Yahoo Finance • atualização a cada 15 minutos • preços ajustados por eventos corporativos")
if not submitted:
    st.info("Digite o código de uma ação e toque em ANALISAR AÇÃO.")
    st.stop()
if not re.fullmatch(r"[A-Z]{4}[0-9]{1,2}(?:\.SA)?", ticker):
    st.warning("Digite um código válido da B3, como VALE3 ou SBSP3.")
    st.stop()
try:
    with st.spinner("Consultando preços e calculando indicadores..."):
        res, df = analyze_ticker(ticker)
except Exception:
    st.error("Não foi possível consultar os dados agora. Tente novamente em alguns minutos.")
    st.stop()
if res is None:
    st.warning("Sem histórico suficiente para esta ação. Confira o código ou tente novamente mais tarde.")
    st.stop()
st.subheader(ticker.removesuffix(".SA"))
st.caption(f"Último pregão disponível: {df.index[-1].strftime('%d/%m/%Y')}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Preço atual", fmt_brl(res["Preço"]))
c2.metric(f"Mínima {date.today().year}", fmt_brl(res["Mínima 2026"]), res["Data mínima"])
c3.metric(f"Máxima {date.today().year}", fmt_brl(res["Máxima 2026"]), res["Data máxima"])
c4.metric("Posição na faixa", fmt_pct(res["Posição na faixa"]))

st.markdown(f"### {res['Classificação']}")
st.write(
    f"**Região do preço:** {res['Região']}  \n"
    f"**Direção recente:** {res['Direção']}  \n"
    f"**Nota de posição do preço:** {res['Nota Preço']}/5  \n"
    f"**Nota de tendência:** {res['Nota Tendência']}/5  \n"
    f"**Nota total:** {res['Nota Total']}/10"
)

a, b, c = st.columns(3)
a.metric("Distância até a mínima", fmt_pct(res["Distância da mínima"]))
b.metric("Distância até a máxima", fmt_pct(res["Potencial até máxima"]))
c.metric("RSI 14", "—" if pd.isna(res["RSI14"]) else f"{res['RSI14']:.1f}")

st.subheader("Indicadores técnicos")
tech = pd.DataFrame({
    "Indicador": ["Preço", "MM20", "MM50", "MM200", "RSI 14", "MACD", "Linha de sinal", "Histograma"],
    "Valor": [
        fmt_brl(res["Preço"]),
        fmt_brl(res["MM20"]),
        fmt_brl(res["MM50"]),
        fmt_brl(res["MM200"]),
        "—" if pd.isna(res["RSI14"]) else f"{res['RSI14']:.2f}",
        "—" if pd.isna(res["MACD"]) else f"{res['MACD']:.4f}",
        "—" if pd.isna(res["MACD Signal"]) else f"{res['MACD Signal']:.4f}",
        "—" if pd.isna(res["MACD Hist"]) else f"{res['MACD Hist']:.4f}",
    ]
})
st.dataframe(tech, use_container_width=True, hide_index=True)

st.subheader(f"Preço em {date.today().year}")
chart_df = df[df.index.year == date.today().year][["Price", "MM20", "MM50"]].rename(columns={"Price": "Preço"})
st.line_chart(chart_df, use_container_width=True)

st.subheader("RSI 14")
st.line_chart(df[["RSI14"]].dropna(), use_container_width=True)

st.subheader("MACD")
st.line_chart(df[["MACD", "MACD_SIGNAL"]].dropna(), use_container_width=True)

st.subheader("Leitura do método")
positive = []
attention = []

if res["Posição na faixa"] <= 35:
    positive.append("Preço em região baixa/interessante da faixa do ano.")
else:
    attention.append("Preço já está na metade superior da faixa do ano.")

if not pd.isna(res["MM20"]) and res["Preço"] > res["MM20"]:
    positive.append("Preço acima da MM20.")
else:
    attention.append("Preço abaixo da MM20.")

if not pd.isna(res["MM50"]) and res["Preço"] > res["MM50"]:
    positive.append("Preço acima da MM50.")
else:
    attention.append("Preço abaixo da MM50.")

if not pd.isna(res["MM20"]) and not pd.isna(res["MM50"]) and res["MM20"] > res["MM50"]:
    positive.append("MM20 acima da MM50.")
else:
    attention.append("MM20 ainda abaixo da MM50.")

if not pd.isna(res["RSI14"]):
    if 50 <= res["RSI14"] <= 70:
        positive.append("RSI mostra força compradora sem estar extremamente esticado.")
    elif res["RSI14"] > 70:
        attention.append("RSI acima de 70: movimento pode estar esticado.")
    else:
        attention.append("RSI abaixo de 50: força compradora ainda fraca.")

if res["MACD"] > res["MACD Signal"]:
    positive.append("MACD acima da linha de sinal.")
else:
    attention.append("MACD abaixo da linha de sinal.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🟢 Sinais favoráveis")
    for item in positive:
        st.write("•", item)
with col2:
    st.markdown("#### ⚠️ Pontos de atenção")
    for item in attention:
        st.write("•", item)

st.caption(
    "Ferramenta educacional. A nota não é recomendação de compra ou venda. "
    "O módulo atual usa preço e indicadores técnicos; fundamentos e notícias podem ser adicionados em uma próxima versão."
)
