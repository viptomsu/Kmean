from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# http://127.0.0.1/add
# http://127.0.0.1/minus
# http://127.0.0.1/multi
# http://127.0.0.1/div

customers = pd.read_csv('./datasets/Mall_Customers.csv')
customers = customers.rename(columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'})

config = {
    "labels": None,
    "k": 0,
    "max_iter": 0,
    "tol": 0.0
}

@app.route('/customers', methods=['GET'] )
@cross_origin(origin='*')
def get_customers():
    page = request.args.get("page")
    responseCustomers = customers.rename(columns={
        "CustomerID": "id",
        "Gender": "gender",
        "Age": "age",
        "Annual_income": "annualIncome",
        "Spending_score": "spendingScore"
    })
    return {"data": responseCustomers.to_dict("records"), "count": len(responseCustomers)}

@app.route("/config", methods=['POST'])
@cross_origin(origins='*')
def post_config_params():
    data = request.get_json()
    k = int(data.get("k"))
    max_iter = int(data.get("max_iter"))
    tol = float(data.get("tol"))
    config["k"] = k
    config["max_iter"] = max_iter
    config["tol"] = tol
    return "Successfully!"

@app.route("/config", methods=['GET'])
@cross_origin(origins='*')
def get_config_params():
    return {"k": config["k"], "max_iter": config["max_iter"], "tol": config["tol"]};

@app.route('/cluster', methods=['GET'] )
@cross_origin(origin='*')
def cluster():
    k = config["k"]
    max_iter = config["max_iter"]
    tol = config["tol"]
    df_3d = customers.drop(columns=(['CustomerID', 'Gender']))
    X = np.array(df_3d.astype(float))

    model = KMeans(n_clusters=k, n_init=10, max_iter=max_iter, tol=tol, random_state=111)
    model.fit(X)
    config["labels"] = model.labels_;
    config["k"] = k;

    customers["label"] = model.labels_;
    return "Successfully!!"

@app.route('/groups', methods=['GET'] )
@cross_origin(origin='*')
def get_groups():
    return list(range(config["k"]))

# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')