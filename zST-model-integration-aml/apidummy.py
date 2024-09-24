from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger
from contextlib import asynccontextmanager
import pandas as pd
import logging
from datetime import datetime
import os
import logging
from logging.config import dictConfig
import math

DEBUG = False
DEMO_MODE = True
if ('DEBUG' in os.environ):
    DEBUG = True if os.environ['DEBUG'] else False

useSnapMl = False
if ('SNAPML' in os.environ):
    DEBUG = True if os.environ['SNAPML'] else False

# Logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG' if DEBUG else 'INFO').upper()
logger.setLevel(LOGLEVEL)
logging.warn(f'LOGLEVEL {LOGLEVEL}')

logging_config = dict(
    version=1,
    formatters={
        'f': {'format':
              '%(asctime)s [%(levelname)s] %(name)s: %(message)s'}
    },
    handlers={
        'h': {'class': 'logging.StreamHandler',
              'formatter': 'f',
              'level': LOGLEVEL}
    },
    root={
        'handlers': ['h'],
        'level': LOGLEVEL,
    },
)
dictConfig(logging_config)

currencies = ["USD", "AED", "SGD", "EURO", "EGP", "INR"]
paymentFormats = ["Cheque", "ACH", "Credit Card", "Wire"]


def format_payment_format(x):
    return paymentFormats[x["paymentFormat"] % len(paymentFormats)]


def format_to_date(x):
    return datetime.fromtimestamp(x['timestamp']).strftime("%Y-%m-%d %H:%M")


def format_currency_received(x):
    return currencies[x["currencyReceived"] % len(currencies)]


def format_currency_sent(x):
    return currencies[x["currencySent"] % len(currencies)]


sources = {}

sources['sourcedf'] = pd.read_csv(
    'data/aml-test-transactions.txt', delim_whitespace=True)


# @asynccontextmanager
# def lifespan(app: FastAPI):
#     logging.info('heavy lifting')
#     # Load the test data
#     sourcedf = pd.read_csv(
#         'data/aml-test-transactions.txt', delim_whitespace=True)
#     sourcedf["timestamp"] = sourcedf.apply(format_to_date, axis=1)
#     sourcedf["currencySent"] = sourcedf.apply(format_currency_sent, axis=1)
#     sourcedf["currencyReceived"] = sourcedf.apply(
#         format_currency_received, axis=1)
#     sourcedf["paymentFormat"] = sourcedf.apply(format_payment_format, axis=1)
#     sources['sourcedf'] = sourcedf
#     logging.info('heavy lifting complete')


app = FastAPI(debug=DEBUG)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/prediction/transaction_list")
async def get_transaction_list(rows: int = 2, page: int = 0):
    if (rows <= 0):
        rows = 2
    sourcedf: pd.DataFrame = sources['sourcedf']
    page = 1+(page % (math.ceil(sourcedf.shape[0]/rows)))
    end = min(sourcedf.shape[0], (page * rows))-1
    start = end + 1 - rows

    logging.info(
        f'shape {sourcedf.shape} pages: {sourcedf.shape[0]/rows}, start: {start}, end: {end}')

    df = sourcedf.loc[start:end, :]
    df.rename(columns={'transactionID': 'id'}, inplace=True)
    df['id'] = df['id'].astype(str)

    result = {'results': [{'id': int(i), 'risk': i, 'prob': 0.5}
                          for i in range(end-start+1)]}
    logging.info(f'shape {len(result["results"])}')

    df['result'] = result['results']

    return df.to_dict(orient="records")