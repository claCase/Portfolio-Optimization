import pandas as pd
import time
from twelvedata import TDClient
import os
import json


def get_tickers_list_SP500():
    import urllib
    from urllib import request
    from bs4 import BeautifulSoup as bso

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Get page
    if not os.path.exists("wiki_tickers_page.txt"):
        get_url = request.urlopen(url).read()
        with open("wiki_tickers_page.txt", "w") as file:
            file.write(str(get_url))
    else:
        with open("wiki_tickers_page.txt", "r") as file:
            get_url = file.read()

    # Get Ticker
    bsdoc = bso(get_url, "html.parser")
    table = bsdoc.find("table", {"id": "constituents"})
    # print(table)
    table = table.find("tbody")
    if not os.path.exists("tickers.txt"):
        # print(get_url)
        a_ = table.findAll("a", {"class": "external text"})
        tickers = []
        for a in a_:
            if a.getText() != "reports":
                tickers.append(str(a.getText()))

        with open("tickers.txt", "w") as file:
            file.write(json.dumps(tickers))
            print("tickers file created")
    else:
        with open("tickers.txt", "r") as file:
            tickers = json.loads(file.read())

    if not os.path.exists("tickers_names.txt"):
        names = []
        trs = table.findAll("tr")
        i = 0
        for tr in trs:
            #print(tr)
            tds = bso.findAll(tr, "td")
            #ais = bso.findAll(tds, "a")
            for i, td in enumerate(tds):
                #print(i, td)
                if i==1:
                    name = bso.find(td, "a").getText()
                    names.append(name)
        with open("tickers_names.txt", "w") as file:
            file.write(json.dumps(names))
    else:
        with open("tickers_names.txt", "r") as file:
            names = json.loads(file.read())

    return tickers, names

def get_tickers_list_PA():
    import urllib
    from urllib import request
    from bs4 import BeautifulSoup as bso
    from string import ascii_uppercase

    alphabet = [letter for letter in ascii_uppercase]
    url = "https://www.borsaitaliana.it/borsa/azioni/listino-a-z.html?initial="

    for letter in alphabet:
        ticker_page = request.urlopen(url + letter).read()
        table = bso(ticker_page, "html.parser").find("table", {"class": "m-table -firstlevel"})
        table_body = table.findAll("tr")

        for row in table_body:
            print(row)
        print(table)


def get_tickers_data(interval="1day", outputsize=5000):
    sp500_tickers, names = get_tickers_list_SP500()
    with open("apikey.txt", "r") as file:
        apikey = json.loads(file.read())

    td = TDClient(apikey=apikey)
    for ticker in sp500_tickers:
        df = td.time_series(symbol=ticker, outputsize=outputsize, interval=interval).as_pandas()
        print(df.head())
        df.to_csv(f"tickers_data/{ticker}.csv")
        time.sleep(10)


def load_data(save_all=True):
    data = pd.DataFrame({"Ticker": [], "datetime": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
    for root, folders, files in os.walk(os.path.join("tickers_data")):
        for file in files:
            print(file)
            data_ticker = pd.read_csv(os.path.join("tickers_data", file))
            ticker = [str(file)[:-4]] * len(data_ticker)
            data_ticker["Ticker"] = ticker
            data = data.append(data_ticker, ignore_index=True)
    if save_all:
        data.to_csv(os.path.join("tickers_data", "all_data.csv"))
    return data
