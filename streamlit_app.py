from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse
from threading import Lock
import re
import pandas as pd
import streamlit as st
import yfinance as yf
from analysis_engine import technical, fundamentals, conclusion, number

st.set_page_config(page_title='Método Mirian — Análise de compra', page_icon='📈', layout='centered')

def safe(call, default):
    try:
        return call()
    except Exception:
        return default

@st.cache_resource
def download_lock():
    return Lock()

@st.cache_data(ttl=900, show_spinner=False)
def history(symbol):
    with download_lock():
        return yf.download(symbol, period='2y', auto_adjust=True, progress=False, threads=False, timeout=15)

@st.cache_data(ttl=21600, show_spinner=False)
def company(symbol):
    stock = yf.Ticker(symbol)
    jobs = {
        'info': (lambda: stock.get_info(), {}),
        'income': (lambda: stock.get_income_stmt(freq='yearly'), pd.DataFrame()),
        'balance': (lambda: stock.get_balance_sheet(freq='yearly'), pd.DataFrame()),
        'cash': (lambda: stock.get_cash_flow(freq='yearly'), pd.DataFrame()),
        'news': (lambda: stock.get_news(count=5), []),
    }
    with ThreadPoolExecutor(max_workers=5) as executor:
        pending = {key: executor.submit(safe, call, default) for key, (call, default) in jobs.items()}
        result = {key: task.result() for key, task in pending.items()}
    result['retrieved'] = datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M UTC')
    return result

def money(value):
    if value is None:
        return 'Não disponível'
    return ('R$ ' + f'{value:,.2f}').replace(',', '_').replace('.', ',').replace('_', '.')

def percent(value):
    return 'Não disponível' if value is None else f'{value:.1f}%'.replace('.', ',')

def compact(value):
    if value is None:
        return 'Não disponível'
    if abs(value) >= 1e9:
        return f'R$ {value / 1e9:.2f} bilhões'.replace('.', ',')
    return f'R$ {value / 1e6:.1f} milhões'.replace('.', ',')

def news_items(items):
    result = []
    for item in items or []:
        content = item.get('content') or item
        url = (content.get('canonicalUrl') or {}).get('url') or content.get('link')
        title = content.get('title')
        if not title or not url or urlparse(url).scheme != 'https':
            continue
        published = content.get('pubDate') or content.get('providerPublishTime')
        try:
            date = pd.to_datetime(published, unit='s', utc=True) if isinstance(published, (int, float)) else pd.to_datetime(published, utc=True)
            if pd.isna(date):
                continue
            age = (pd.Timestamp.now(tz='UTC') - date).total_seconds() / 86400
            if not 0 <= age <= 30:
                continue
        except Exception:
            continue
        provider = content.get('provider') or {}
        source = provider.get('displayName', 'Fonte não identificada') if isinstance(provider, dict) else str(provider)
        result.append((title, url, date.strftime('%d/%m/%Y'), source))
    return result[:5]

st.title('📈 Método Mirian')
st.write('Entenda o que favorece a compra, o que preocupa e o que ainda falta confirmar.')
st.caption('Análise para acompanhar movimentos de semanas a meses • versão 2.0')
with st.form('search'):
    ticker = st.text_input('Qual ação você quer analisar?', value='SBSP3', placeholder='Ex.: VALE3 ou PETR4').strip().upper()
    submitted = st.form_submit_button('ANALISAR COMPRA', type='primary', width='stretch')
if submitted:
    if not re.fullmatch(r'[A-Z]{4}[0-9]{1,2}(?:\.SA)?', ticker):
        st.warning('Digite um código de ação da B3, como VALE3 ou SBSP3.')
        st.stop()
    st.session_state['selected'] = ticker.removesuffix('.SA')
if 'selected' not in st.session_state:
    st.info('Digite a ação para ver uma conclusão explicada em linguagem simples.')
    st.caption('A análise considera preços, balanços disponíveis e Ibovespa. Fatos relevantes e preço justo ainda exigem verificação; não emitimos sinal automático de compra nesta versão.')
    st.stop()

ticker = st.session_state['selected']
symbol = ticker + '.SA'
with st.spinner('Consultando preços, balanços e contexto do mercado...'):
    with ThreadPoolExecutor(max_workers=3) as executor:
        stock_job = executor.submit(safe, lambda: history(symbol), pd.DataFrame())
        market_job = executor.submit(safe, lambda: history('^BVSP'), pd.DataFrame())
        company_job = executor.submit(company, symbol)
        df, market, data = stock_job.result(), market_job.result(), company_job.result()
    tech = technical(df, market)
    info = data['info'] or {}
    fund = fundamentals(info, data['income'], data['balance'], data['cash'])

st.subheader(ticker)
st.text(info.get('longName') or info.get('shortName') or 'Nome da empresa não disponível')
if info.get('quoteType') and info['quoteType'] != 'EQUITY':
    st.warning('Esta análise foi preparada para ações de empresas. O ativo informado exige outro modelo.')
    st.stop()
pe, pb = number(info.get('trailingPE')), number(info.get('priceToBook'))
valuation = 'Preço justo ainda não determinado'
title, explanation = conclusion(tech, fund, valuation)
if title.startswith('🔴'):
    st.error(f'**{title}**\n\n{explanation}')
else:
    st.warning(f'**{title}**\n\n{explanation}')
if not tech['available']:
    st.caption('Fonte: Yahoo Finance. A consulta pode estar temporariamente indisponível.')
    st.stop()
st.caption(f"Último fechamento: {tech['date'].strftime('%d/%m/%Y')} • {money(tech['price'])} • não é cotação ao vivo")
market_label = 'Favorável pelos filtros de tendência' if tech['market_ok'] is True else 'Não confirma uma tendência favorável' if tech['market_ok'] is False else 'Dados insuficientes'
st.table(pd.DataFrame([
    {'O que avaliamos': 'Empresa', 'Leitura': fund['state']},
    {'O que avaliamos': 'Preço', 'Leitura': valuation},
    {'O que avaliamos': 'Momento', 'Leitura': tech['stage']},
    {'O que avaliamos': 'Bolsa brasileira', 'Leitura': market_label},
    {'O que avaliamos': 'Risco da operação', 'Leitura': 'Elevado pelos filtros' if tech['risk_high'] else 'Exige controle; sem alerta elevado nos filtros'},
    {'O que avaliamos': 'Setor e fatos relevantes', 'Leitura': 'Verificação ainda pendente'},
]))
st.markdown('### Por que chegamos a essa conclusão?')
reasons = list(fund['reasons'])
if tech['down']:
    reasons.insert(0, 'O preço está abaixo das médias de médio e longo prazo, e a média longa está caindo.')
elif tech['up']:
    reasons.append('O preço e as médias mostram uma tendência de alta pelos filtros definidos.')
else:
    reasons.append('A melhora recente ainda não confirmou uma tendência de alta consistente.')
if fund['gaps']:
    reasons.append(fund['gaps'][0])
elif fund['profit'] is not None and fund['profit'] > 0 and fund['flow'] is not None and fund['flow'] > 0:
    reasons.append('O último exercício disponível apresentou lucro e geração de caixa operacional positivos.')
if tech['market_ok'] is False:
    reasons.append('O Ibovespa ainda não confirma uma tendência favorável pelos filtros de longo prazo.')
if tech['relative'] is not None:
    lead = 'acima' if tech['relative'] >= 0 else 'abaixo'
    reasons.append(f"Nos últimos 63 pregões em comum, a ação rendeu {abs(tech['relative']):.1f} pontos percentuais {lead} do Ibovespa.")
for reason in reasons[:3]:
    st.write('• ' + reason)
st.markdown('### Principal risco')
if fund['reasons']:
    st.write(fund['reasons'][0])
elif tech['turnover'] is None or tech['turnover'] < 2_000_000:
    st.write('O volume financeiro é baixo ou desconhecido. Pode ser mais difícil vender pelo preço esperado.')
elif tech['down']:
    st.write('A queda pode continuar mesmo depois de uma recuperação curta.')
elif tech['volatility'] > 45:
    st.write('O preço tem oscilado intensamente. Uma mudança rápida pode causar uma perda relevante.')
else:
    st.write('A recuperação pode falhar. Balanços e acontecimentos ainda não verificados podem mudar a análise.')
st.markdown('### O que falta para considerar uma compra?')
pending = []
if not tech['up']:
    pending.append('Confirmar uma tendência de alta com preço acima das médias de 50 e 200 pregões e médias subindo.')
if tech['market_ok'] is not True:
    pending.append('Verificar a melhora da tendência do Ibovespa.')
if not tech['buying_volume']:
    pending.append('Observar participação maior de compradores: alta acompanhada de volume acima da média anterior.')
pending.extend(fund['gaps'])
pending.extend(['Avaliar o preço em relação a empresas comparáveis e às perspectivas do negócio.', 'Verificar os últimos resultados trimestrais, fatos relevantes e condições do setor.'])
for item in pending:
    st.write('• ' + item)

with st.expander('Empresa: lucros, caixa e dívidas'):
    st.write('Leitura inicial de demonstrações anuais fornecidas pelo Yahoo Finance. Não substitui a revisão dos últimos balanços trimestrais e das notas explicativas.')
    if fund['date'] is not None:
        st.caption('Exercício mais recente encontrado: ' + fund['date'].strftime('%d/%m/%Y'))
    st.write('**Lucro anual:** ' + compact(fund['profit']))
    st.write('**Caixa gerado pela operação:** ' + compact(fund['flow']))
    st.write('**Variação anual da receita:** ' + percent(fund['growth']))
    ratio = fund['debt_equity']
    st.write('**Dívida total / patrimônio:** ' + (f'{ratio:.2f} vezes' if ratio is not None else 'Não disponível'))
    st.caption('A dívida exige interpretação por setor. Esta relação não mede sozinha a capacidade de pagamento e não é aplicada como filtro a bancos e seguradoras.')
    for gap in fund['gaps']:
        st.warning(gap)
with st.expander('Preço: está barato ou apenas caiu?'):
    st.write('A mínima do ano não é preço justo. Não classificamos uma ação como barata apenas por ter caído.')
    st.write('**Preço / lucro dos últimos 12 meses:** ' + (f'{pe:.1f} vezes' if pe is not None and pe > 0 else 'Não disponível ou não interpretável'))
    st.write('**Preço / valor patrimonial:** ' + (f'{pb:.1f} vezes' if pb is not None and pb > 0 else 'Não disponível ou não interpretável'))
    st.caption('Múltiplos são referências, não aprovação de compra. Exigem comparação por setor, qualidade dos lucros e crescimento. A data-base destes múltiplos não é garantida pelo provedor.')
    st.write(f"**Faixa dos últimos 12 meses:** {money(tech['low'])} a {money(tech['high'])}.")
    st.caption('Preços históricos ajustados por eventos corporativos. A máxima passada não é um alvo de retorno.')
with st.expander('Risco: entenda os níveis de referência'):
    st.write(f"**Fundo dos 20 pregões anteriores:** {money(tech['support'])}.")
    st.write(f"**Topo dos 60 pregões anteriores:** {money(tech['resistance'])}.")
    if tech['loss'] > 0:
        st.write(f"Uma queda do último fechamento até esse fundo seria de **{percent(tech['loss'])}**.")
    else:
        st.write('O preço já está no fundo recente ou abaixo dele; esse nível não serve como proteção abaixo da entrada.')
    if tech['rr'] is not None:
        st.write(f"A distância até o topo é **{tech['rr']:.1f} vezes** a distância até o fundo.")
    else:
        st.write('Não há relação válida entre essas duas distâncias para a situação atual.')
    st.caption('São extremos históricos, não suporte ou resistência confirmados, ordem de stop, preço-alvo ou limite de perda. Saltos de preço e custos podem aumentar a perda real.')
    st.write('**Volume financeiro diário médio aproximado:** ' + compact(tech['turnover']))
    st.write('**Oscilação anualizada estimada pelos últimos 60 pregões:** ' + percent(tech['volatility']))
with st.expander('Notícias e fontes para verificar o contexto'):
    st.write('Manchetes do Yahoo Finance podem tratar do mercado em geral. Não são interpretadas automaticamente como positivas ou negativas e não substituem os fatos relevantes da empresa.')
    news = news_items(data['news'])
    if not news:
        st.info('Nenhuma notícia recente com fonte e data utilizáveis foi recebida. Isso não significa ausência de acontecimentos importantes.')
    for headline, url, date, provider in news:
        st.link_button(headline, url)
        st.caption(f'{provider} • {date}')
    st.link_button('Consultar documentos oficiais na CVM', 'https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx')
    st.link_button('Consultar a ação no Yahoo Finance', f'https://finance.yahoo.com/quote/{symbol}/')
    st.caption('Dados da empresa consultados em ' + data['retrieved'])
with st.expander('Ver gráfico e entender os critérios'):
    st.line_chart(tech['chart'], width='stretch')
    st.write('Médias móveis são médias dos fechamentos. O filtro de alta exige preço acima da média de 50 pregões, média de 50 acima da de 200, e ambas maiores que 20 pregões antes.')
    st.write('A comparação com o Ibovespa usa 63 pregões em comum. Compara retorno e não é o indicador RSI.')
    st.write('O filtro de risco acende com oscilação anualizada acima de 45%, volume financeiro médio abaixo de R$ 2 milhões ou distância até o fundo recente acima de 12%. São regras iniciais, não padrões universais.')
    st.write('Não existe mais uma soma de pontos que permita ao preço baixo compensar problemas importantes. Volume elevado só confirma compradores se o fechamento subir.')
    st.info('As regras ainda não passaram por validação histórica de estratégia com custos, ativos deslistados e períodos fora da amostra. Por isso, sinais positivos continuam exigindo confirmação e esta versão não emite aprovação automática de compra.')
st.caption('Ferramenta de triagem educacional. Não avalia sua carteira, seu perfil ou sua capacidade de suportar perdas. Preços em cache por até 15 minutos; dados empresariais por até 6 horas. Atualizar a consulta não transforma o fechamento diário em cotação ao vivo.')
