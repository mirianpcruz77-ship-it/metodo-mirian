import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_engine import technical, fundamentals, conclusion
from streamlit.testing.v1 import AppTest

TODAY = pd.Timestamp.today().normalize()
def frame(up=True):
    idx = pd.bdate_range(end=TODAY, periods=400)
    p = np.linspace(20, 40, 400) if up else np.linspace(40, 20, 400)
    return pd.DataFrame({'Close': p, 'High': p + .5, 'Low': p - .5, 'Volume': 1_000_000}, index=idx)

def statements():
    years = [pd.Timestamp(TODAY.year - n, 12, 31) for n in (1, 2, 3)]
    income = pd.DataFrame([[100, 90, 80], [1000, 950, 900]], index=['NetIncome', 'TotalRevenue'], columns=years)
    balance = pd.DataFrame([[150, 150, 150], [500, 500, 500]], index=['TotalDebt', 'StockholdersEquity'], columns=years)
    cash = pd.DataFrame([[120, 110, 100]], index=['OperatingCashFlow'], columns=years)
    return income, balance, cash

INFO = {'sector': 'Utilities', 'currency': 'BRL', 'financialCurrency': 'BRL', 'quoteType': 'EQUITY'}
class Rules(unittest.TestCase):
    def test_sell_volume_is_not_buy_confirmation(self):
        d = frame()
        d.iloc[-1, d.columns.get_loc('Close')] = d.Close.iloc[-2] - .1
        d.iloc[-1, d.columns.get_loc('Volume')] = 4_000_000
        self.assertFalse(technical(d, frame())['buying_volume'])

    def test_missing_and_stale_prices(self):
        self.assertFalse(technical(frame().head(30), frame())['available'])
        stale = frame(); stale.index -= pd.Timedelta(days=30)
        self.assertFalse(technical(stale, frame())['available'])

    def test_loss_denominator(self):
        result = technical(frame(), frame())
        self.assertAlmostEqual(result['loss'], (result['price'] - result['support']) / result['price'] * 100)

    def test_missing_data_and_banks_block_approval(self):
        missing = fundamentals({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.assertTrue(missing['gaps'])
        self.assertIn('INCOMPLETA', conclusion(technical(frame(), frame()), missing, '')[0])
        bank = fundamentals(dict(INFO, sector='Financial Services'), *statements())
        self.assertTrue(bank['gaps'])

    def test_loss_and_downtrend_override_price(self):
        income, balance, cash = statements()
        income.iloc[0, 0] = -10
        fund = fundamentals(INFO, income, balance, cash)
        self.assertIn('DESFAVORÁVEL', conclusion(technical(frame(), frame()), fund, '')[0])
        fund = fundamentals(INFO, *statements())
        self.assertIn('DESFAVORÁVEL', conclusion(technical(frame(False), frame()), fund, '')[0])

    def test_unvalidated_never_approves(self):
        fund = fundamentals(INFO, *statements())
        self.assertFalse(fund['gaps'])
        tech = technical(frame(), frame())
        tech['relative'] = 1.0
        self.assertIn('NÃO CONFIRMADA', conclusion(tech, fund, '')[0])

    def test_ui(self):
        app = str(Path(__file__).with_name('streamlit_app.py'))
        at = AppTest.from_file(app, default_timeout=45).run()
        self.assertFalse(at.exception)
        at.text_input[0].set_value('???'); at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(at.warning)
        stock = MagicMock()
        stock.get_info.return_value = INFO
        stock.get_income_stmt.return_value, stock.get_balance_sheet.return_value, stock.get_cash_flow.return_value = statements()
        stock.get_news.return_value = []
        with patch('yfinance.download', return_value=frame()), patch('yfinance.Ticker', return_value=stock):
            at.text_input[0].set_value('VALE3'); at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.expander), 5)
        self.assertIn('AGUARDAR CONFIRMAÇÃO', at.warning[0].value)
        self.assertEqual(len(at.table[0].value), 6)

if __name__ == '__main__':
    unittest.main(verbosity=2)
