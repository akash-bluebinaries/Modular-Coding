import sys
from flask import Flask
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():

    try:
        raise Exception("Trial of exception module")
    except Exception as e:
        ml = CustomException(e,sys)
        logging.info(ml.error_message)
    

    logging.info("Testing exception")

    return "Welcome to testing exception"


if __name__ == "__main__":
    app.run(debug = True) # 5000