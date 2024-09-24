from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger
from contextlib import asynccontextmanager
import math

import os
import os.path
from datetime import datetime, timedelta
import io
import shap
import pandas as pd
import numpy as np
import logging
from logging.config import dictConfig
from src.wmlz_client import AmlWmlzClient
from src.inmemory_model import InMemoryModel
from src.gp_preprocessing import GpPreprocessing
import pickle
import matplotlib.pyplot as plt
import matplotlib
from dotenv import dotenv_values

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
    now = datetime.now() - timedelta(1)
    return datetime.fromtimestamp(x['timestamp']).replace(day=now.day, month=now.month, year=now.year).strftime("%Y-%m-%d %H:%M:%S")


def format_currency_received(x):
    return currencies[x["currencyReceived"] % len(currencies)]


def format_currency_sent(x):
    return currencies[x["currencySent"] % len(currencies)]


sources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    use_snapml = False
    logging.info('heavy lifting')

    config = {
        **dotenv_values(".env"),  # load shared development variables
        **os.environ,  # override loaded values with environment variables
    }
    # 'data/transformed.npy'
    logging.debug(f'configuration loaded {config}')

    shap_path = 'data/shapvalues.pkl'
    if ('SHAPPATH' in config):
        shap_path = config['SHAPPATH']
    test_transaction = 'data/aml-test-transactions.txt'
    if ('TESTPATH' in config):
        test_transaction = config['TESTPATH']
    errorMsg = None
    use_wmlz = False
    if ('USE_WMLZ' in config):
        use_wmlz = True
        if (not ('WML_AUTH_URL' in config and 'WML_USER' in config and 'WML_PASS' in config and 'WMLZ_SCORING_URL' in config)):
            errorMsg = 'WMLz Credentials configuration are missing'
        if ('WMLZ_SCORING_URL' not in config):
            errorMsg = 'WMLz URL configuration are missing'
    else:
        if ('USE_SNAPML' in config):
            use_snapml = True
    sources['USE_WMLZ'] = use_wmlz

    if ('TRANSFORMED_DATA_PATH' not in config):
        config['TRANSFORMED_DATA_PATH'] = 'data/transformed.npy'
    logging.info('configuration is validated')
    if (errorMsg is not None and len(errorMsg) > 0):
        logging.error(errorMsg)
        raise Exception(errorMsg)

    # Load the test data
    sourcedf = pd.read_csv(test_transaction, delim_whitespace=True)
    sourcedf["timestamp"] = sourcedf.apply(format_to_date, axis=1)
    sourcedf["currencySent"] = sourcedf.apply(format_currency_sent, axis=1)
    sourcedf["currencyReceived"] = sourcedf.apply(
        format_currency_received, axis=1)
    sourcedf["paymentFormat"] = sourcedf.apply(format_payment_format, axis=1)
    sources['sourcedf'] = sourcedf

    try:
        logging.info('loading gp pre-processing')
        sources['gpPreprocessing'] = GpPreprocessing(
            config['TRANSFORMED_DATA_PATH'])
    except:
        logging.exception('Error while loading GP pre-processing')

    if ('USE_WMLZ' in config):
        try:
            logging.info('loading wmlz client')
            sources['wmlzclient'] = AmlWmlzClient(auth_url=config['WML_AUTH_URL'], username=config['WML_USER'],
                                                  password=config['WML_PASS'], scoring_url=config['WMLZ_SCORING_URL'])
        except:
            logging.exception('Error while loading WMLz client')
    else:
        try:
            logging.info('loading inmemory model')
            sources['inmemory_model'] = InMemoryModel(use_snap_ml=use_snapml)
        except:
            logging.exception('Error while loading immemory model')

    try:
        logging.info('loading shap explainability')
        # Load explanations from the saved .pkl file
        with open(shap_path, 'rb') as input_file:
            sources['shap_values'] = pickle.load(input_file)
    except:
        logging.exception('Error while loading shap values')

    logging.info('heavy lifting complete')

    yield
    logging.info('cleaning up resource')
    # Clean ups and release the resources
    sources.clear()

app = FastAPI(lifespan=lifespan, debug=DEBUG)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


@app.get("/transaction_list")
async def get_transaction_list(rows: int = 12, page: int = 0):
    if (rows <= 0):
        rows = 2
    sourcedf: pd.DataFrame = sources['sourcedf']
    logging.debug(f'total pages {sourcedf.shape[0]/rows}')
    page = 1+(page % (math.ceil(sourcedf.shape[0]/rows)))
    end = (page * rows)-1
    start = end + 1 - rows

    df = sourcedf.loc[start:end, :]
    df.rename(columns={'transactionID': 'id'}, inplace=True)
    df['id'] = df['id'].astype(str)

    return df.to_dict(orient="records")


@app.post("/predict")
async def get_predict(ids: list[int]):
    sourcedf: pd.DataFrame = sources['sourcedf']
    filtered = sourcedf.loc[sourcedf['transactionID'].isin(ids)]
    logging.debug(f"filtered rows {filtered.shape}")
    # get index
    selected_indexes = list(filtered.index)
    logging.debug(f'selected indexes {selected_indexes}. selected ids {ids}')

    gpPreprocessing: GpPreprocessing = sources['gpPreprocessing']
    (feature_out, elapsed_time) = gpPreprocessing.preprocess(selected_indexes)
    logging.info(
        f'pre-processed {len(selected_indexes)} items in {elapsed_time}')
    logging.debug(feature_out)
    if (sources['USE_WMLZ']):
        wmlzclient: AmlWmlzClient = sources['wmlzclient']
        results = wmlzclient.get_prediction(feature_out[:, 1:])
    else:
        inmemory_model: InMemoryModel = sources['inmemory_model']
        results = inmemory_model.get_prediction(feature_out[:, 1:])
    logging.info(f'Prediction result {results}')
    return results


@app.get("/prediction/transaction_list")
async def get_pred_transaction_list(rows: int = 12, page: int = 0):
    if (rows <= 0):
        rows = 2
    sourcedf: pd.DataFrame = sources['sourcedf']
    logging.info(f'total pages {sourcedf.shape[0]/rows}')
    page = 1+(page % (math.ceil(sourcedf.shape[0]/rows)))
    end = min(sourcedf.shape[0], (page * rows))-1
    start = end + 1 - rows

    df = sourcedf.loc[start:end, :]
    df.rename(columns={'transactionID': 'id'}, inplace=True)
    df['id'] = df['id'].astype(str)

    gpPreprocessing: GpPreprocessing = sources['gpPreprocessing']
    (feature_out, elapsed_time) = gpPreprocessing.preprocess(list(df.index))
    logging.info(
        f'pre-processed {df.shape} items in {elapsed_time}')
    logging.debug(feature_out)

    if (sources['USE_WMLZ']):
        wmlzclient: AmlWmlzClient = sources['wmlzclient']
        results = wmlzclient.get_prediction(feature_out[:, 1:])
    else:
        inmemory_model: InMemoryModel = sources['inmemory_model']
        results = inmemory_model.get_prediction(feature_out[:, 1:])

    logging.debug(results)
    df['result'] = [
        {'id': id, 'risk': 1 if probability >= 0.5 else 0, 'prob': probability} for probability, id in zip(results, df['id'])]
    return df.to_dict(orient="records")


@app.get("/explain")
async def get_explain(background_tasks: BackgroundTasks, id: int):
    sourcedf: pd.DataFrame = sources['sourcedf']
    samples = sourcedf.index[sourcedf['transactionID'].isin([id])].tolist()
    logger.info("Numbers in samples are: {}".format(
        ' '.join(map(str, samples))))

    img_buf = create_img(samples)
    # get the entire buffer content
    # because of the async, this will await the loading of all content
    bufContents: bytes = img_buf.getvalue()
    background_tasks.add_task(img_buf.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return Response(bufContents, headers=headers, media_type='image/png')


def create_img(samples):
    img_buf = io.BytesIO()
    explanations = sources['shap_values']
    text_color = "#f4f4f4"
    for row_id in samples:
        fig = plt.figure(row_id)
        ax0 = fig.add_subplot(111)
        shap.plots.bar(explanations[row_id], max_display=8, show=False)
        ax0.tick_params(axis='x', labelsize=9, colors=text_color)
        ax0.tick_params(axis='y', labelsize=12, colors=text_color)

        plt.gcf().set_size_inches(9, 5)
        plt.title("Most Impacting Features", color=text_color)
        plt.tight_layout()
        plt.xlabel("Level of Impact")
        # ax0.spines['bottom'].set_color(text_color)
        ax0.xaxis.label.set_color(text_color)
        customize_color(plt.gcf())
        plt.savefig(img_buf, format='png', transparent=True)
        plt.close()
    return img_buf


def customize_color(figure):
    # Default SHAP colors
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"
    # Custom colors
    positive_color = "#c00000"
    negative_color = "#008bfb"
    text_color = '#e0e0e0'
    # Change the colormap of the artists
    for fc in figure.get_children():
        # Ignore last Rectangle
        for fcc in fc.get_children()[:-1]:
            if (isinstance(fcc, matplotlib.patches.Rectangle)):
                if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                    fcc.set_facecolor(positive_color)
                elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                    fcc.set_color(negative_color)
            elif (isinstance(fcc, plt.Text)):
                if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color or matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                    fcc.set_color(text_color)
