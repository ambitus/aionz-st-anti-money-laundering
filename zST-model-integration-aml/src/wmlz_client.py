import requests
import time
import json
import logging
import numpy as np


class AmlWmlzClient:
    def __init__(self, auth_url, username, password, scoring_url):
        self.auth_url = auth_url
        self.username = username
        self.password = password
        self.scoring_url = scoring_url
        self.auth_token = self.get_authentication()
        logging.info("Authentication token loaded")

    def get_authentication(self):
        payload = json.dumps({
            "username": self.username,
            "password": self.password
        })
        headers = {
            "Content-Type": "application/json",
            "Control": "no-cache"
        }
        logging.info('Authentication token generation')
        try:
            response = requests.request(
                "POST", self.auth_url, headers=headers, data=payload, verify=False)
            auth_response_json = response.json()
            auth_token = auth_response_json["token"]
        except:
            raise InvalidWMLzPasswordException
        return auth_token

    def get_prediction(self, rows: np.ndarray):
        data = [{f'x{i+1}': value for i,
                 value in enumerate(row_data)} for row_data in rows]
        logging.debug(data)
        url = self.scoring_url
        payload = json.dumps(data)
        try:
            token = self.auth_token
        except InvalidWMLzPasswordException:
            raise InvalidWMLzPasswordException
        headers = {
            "Authorization": "Bearer %s" % token,
            "Content-Type": "application/json"
        }
        try:
            tt0 = time.time()
            response = requests.request(
                "POST", url, headers=headers, data=payload, verify=False)
            x = response.json()
            tt1 = time.time()
            logging.info(f'Inference {rows.shape[0]} items in {tt1-tt0}')
            if not isinstance(x, list) and (x["code"] == "WML_OS_0015" or x["code"] == "WML_OS_0016"):
                logging.info('inference call failed. retrying with new token')
                token = self.get_authentication()
                headers = {
                    "Authorization": "Bearer %s" % token,
                    "Content-Type": "application/json"
                }
                tt0 = time.time()
                response = requests.request(
                    "POST", url, headers=headers, data=payload, verify=False)
                tt1 = time.time()
                logging.info(f'Inference {rows.shape[0]} items in {tt1-tt0}')
        except:
            logging.error('inference call error')
        results = response.json()
        results_aml = [result['probability(1)'] for result in results]
        return results_aml


class InvalidWMLzPasswordException(Exception):
    """Raised when the WMLz Password has been expired"""
    pass
