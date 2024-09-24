# Copyright 2024 IBM Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from snapml import GraphFeaturePreprocessor
import numpy as np
import pandas as pd
from datetime import datetime
import os
import csv
import time

# Raw input transactions
raw_transaction_path = "./aml-demo-data/HI-Small_Trans.csv"
formatted_data_path = "./aml-demo-data/out_dir/"

# Formatted transactions
out_path = formatted_data_path + "formatted_transactions.csv"

# Formatted test, train, and transactions to be preloaded
n_test = 863901
n_preload = 1000000
test_trans_path = formatted_data_path + "aml-hi-small_test_trans.csv"
train_trans_path = formatted_data_path + "aml-hi-small_train_trans.csv"
test_labels_path = formatted_data_path + "aml-hi-small_test_labels.csv"
train_labels_path = formatted_data_path + "aml-hi-small_train_labels.csv"
preload_trans_path = formatted_data_path + "aml-hi-small_preload_trans.csv"

# Formatting information
acc_map_path = formatted_data_path + "account_mapping.csv"
format_map_path = formatted_data_path + "payment_format_mapping.csv"
currency_map_path = formatted_data_path + "currency_format_mapping.csv"

#######################################################################################################

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

## Function used to map categorical value to an integer
def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

## Function to convert a currency to dollars
def convert_to_dolars(amount, currency):
    exchange_rate = {
        "US Dollar":1.0,
        "Euro":0.8534,
        "Yuan":6.6976,
        "Yen":105.4,
        "Rupee":73.444,
        "Ruble":77.804,
        "UK Pound":0.7742,
        "Canadian Dollar":1.3193,
        "Australian Dollar":1.4128,
        "Mexican Peso":21.1431,
        "Brazil Real":5.6465,
        "Swiss Franc":0.9150,
        "Shekel":3.3770,
        "Saudi Riyal":3.7511,
        "Bitcoin":0.0000841611 
    }
    
    if currency in exchange_rate:
        return float(amount)/exchange_rate[currency]
    else:
        print("Currency ", currency, " is not valid")
        exit(1)

## Utility function
def print_dict_to_csv(d, filename):
    with open(filename, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in d.items():
            writer.writerow([key, value])
            
#######################################################################################################

# load raw transactions
raw = pd.read_csv(raw_transaction_path)

formatted_transactions = []
account_mapping = []

n_rows = len(raw)

firstTs = None

t0 = time.time()

## Formatting transactions
for i in range(n_rows):
    if (i+1) % 50000 == 0:
        print("Processed ", i+1 ," transactions", flush=True)
    dt = datetime.strptime(raw['Timestamp'].iloc[i], '%Y/%m/%d %H:%M')

    if firstTs is None:
        startTime = datetime(dt.year, dt.month, dt.day)
        firstTs = startTime.timestamp() - 10

    ts = dt.timestamp() - firstTs

    cur1 = get_dict_val(raw["Receiving Currency"].iloc[i], currency)
    cur2 = get_dict_val(raw["Payment Currency"].iloc[i], currency)

    fmt = get_dict_val(raw["Payment Format"].iloc[i], paymentFormat)

    bank1 = get_dict_val(raw["From Bank"].iloc[i], bankAcc)
    bank2 = get_dict_val(raw["To Bank"].iloc[i], bankAcc)

    for raw_acc in [raw.iloc[i,2], raw.iloc[i,4]]:
        if raw_acc not in account:
            val = len(account)
            account[raw_acc] = val
            account_mapping.append(
                {
                    "Original" : raw_acc,
                    "Mapped" : val
                }
            )

    fromId = get_dict_val(raw.iloc[i,2], account)
    toId = get_dict_val(raw.iloc[i,4], account)

    amountReceivedUsd = convert_to_dolars(raw["Amount Received"].iloc[i], raw["Receiving Currency"].iloc[i])
    amountPaidUsd = convert_to_dolars(raw["Amount Paid"].iloc[i], raw["Payment Currency"].iloc[i])

    transaction = {
        "EdgeID" : i, 
        "SourceAccountId" : fromId,
        "TargetAccountId" : toId,
        "Timestamp" : ts,
        "Amount Received" : raw["Amount Received"].iloc[i],
        "Receiving Currency" : cur1,
        "Amount Received [USD]" : amountReceivedUsd,
        "Amount Paid" : raw["Amount Paid"].iloc[i],
        "Payment Currency" : cur2,
        "Amount Paid [USD]" : amountPaidUsd,
        "SourceBankId" : bank1, 
        "TargetBankId" : bank2, 
        "Payment Format" : fmt,
        "Year" : dt.year,
        "Month" : dt.month,
        "Day" : dt.day,
        "Hour" : dt.hour,
        "Minute" : dt.minute,
        "Is Laundering" : raw["Is Laundering"].iloc[i]
    }
    
    formatted_transactions.append(transaction)
    
t1 = time.time()
    
print("Done in ", (t1-t0), ' s')
    
# printing currency and payment format mapping
print_dict_to_csv(currency, currency_map_path)
print(currency)

print_dict_to_csv(paymentFormat, format_map_path)
print(paymentFormat)

# Printing account mapping
df_acc_map = pd.DataFrame.from_dict(account_mapping) 
df_acc_map.to_csv(acc_map_path, index=False)

# printing formatted transactions
df_trans = pd.DataFrame.from_dict(formatted_transactions)
df_trans = df_trans.sort_values(by=['Timestamp', 'EdgeID'])
df_trans.to_csv(out_path, index=False)

# Dividing into train, test, and preload transactions
df_test = df_trans[-n_test:]
df_train = df_trans[:-n_test]
df_preload = df_train[-n_preload:]

# Extract labels
df_test_labels = df_test["Is Laundering"]
df_train_labels = df_test["Is Laundering"]

# Drop labels
df_test = df_test.drop('Is Laundering', axis=1)
df_train = df_train.drop('Is Laundering', axis=1)
df_preload = df_preload.drop('Is Laundering', axis=1)

# Write the files
df_test.to_csv(test_trans_path, index=False)
df_train.to_csv(train_trans_path, index=False)
df_preload.to_csv(preload_trans_path, index=False)
df_test_labels.to_csv(test_labels_path, index=False)
df_train_labels.to_csv(train_labels_path, index=False)

t2 = time.time()

print("Done in ", (t2-t0), ' s')