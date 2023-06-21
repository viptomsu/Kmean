from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request

import json

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Đọc dữ liệu và gán lại tên cột cho dễ xử lý
customers = pd.read_csv('./datasets/Mall_Customers.csv')
customers = customers.rename(
    columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'})

# khởi tạo labels setting
label = {
    "labels": [],
}
# khởi tạo config để lưu lại các setting cho thuật toán sau này
config = {
    "range_start": 0,
    "range_end": 0,
    "k": 0,
    "max_iter": 0,
    "tol": 0.0
}
# Mã màu mặc định cho các nhóm
default_colors = ['#FF0000', '#00FF00', '#0000FF', '#ffcc00', '#FF00FF', '#00FFFF', '#FFA500']

@app.route('/customers', methods=['GET'])
@cross_origin(origins='*')
def get_customers():
    page = request.args.get("page")
    response_customers = customers.rename(columns={
        "CustomerID": "id",
        "Gender": "gender",
        "Age": "age",
        "Annual_income": "annualIncome",
        "Spending_score": "spendingScore"
    })
    return {"data": response_customers.to_dict("records"), "count": len(response_customers)}


@app.route("/config", methods=['POST'])
@cross_origin(origins='*')
def post_config_params():
    data = request.get_json()
    max_iter = int(data.get("max_iter"))
    tol = float(data.get("tol"))
    config["range_start"] = int(data.get("range_start"))
    config["range_end"] = int(data.get("range_end"))
    config["max_iter"] = max_iter
    config["tol"] = tol
    return "Successfully!"

@app.route("/config/k", methods=["POST"])
@cross_origin(origins='*')
def config_k():
    config["k"] = int(request.get_json().get("k"))
    return "Successfully!"

@app.route("/config", methods=['GET'])
@cross_origin(origins='*')
def get_config_params():
    return json.dumps(config)


@app.route('/cluster', methods=['GET'])
@cross_origin(origins='*')
def cluster():
    if config["k"] > 1:
        k = config["k"]
        max_iter = config["max_iter"]
        tol = config["tol"]
        df_3d = customers.drop(columns=(['CustomerID', 'Gender']))
        X = np.array(df_3d.astype(float))

        model = KMeans(n_clusters=k, n_init=10, max_iter=max_iter, tol=tol, random_state=111)
        model.fit(X)
        for i in range(k):
            item = {
                'id': i,
                'name': '',
                'desc': '',
                'color': default_colors[i % len(default_colors)]
            }
            label["labels"].append(item)

        config["k"] = k
        customers["label"] = model.labels_
        return "Successfully!!"
    else:
        return "Please provide params!!s"


@app.route('/groups', methods=['GET'])
@cross_origin(origins='*')
def get_groups():
    return label["labels"]

@app.route('/groups', methods=['POST'])
@cross_origin(origins='*')
def config_groups():
    data = request.get_json()
    label["labels"] = data
    return "Successfully!!"

@app.route('/circle-chart', methods=['GET'] )
@cross_origin(origin='*')
def get_groups_chart():
    if 'label' in customers.columns:
        label_counts = customers['label'].value_counts().reset_index()
        label_counts.columns = ['label', 'value']
        label_counts['percentage'] = label_counts['value'] / len(customers) * 100

        color_name_dict = {item['id']: (item['color'], item['name']) for item in label["labels"]}
        label_counts[['color', 'name']] = label_counts['label'].map(color_name_dict).apply(pd.Series)

        return label_counts.to_json(orient='records')
    else:
        return []

@app.route("/k-chart", methods=["GET"])
@cross_origin(origins='*')
def get_k_chart_data():
    if config["max_iter"] > 0 and config["tol"] > 0 and config["range_start"] > 1 and config["range_end"] > 1:
        inertia = []
        df_3d = customers.drop(columns=(['CustomerID', 'Gender']))
        X = np.array(df_3d.astype(float))
        for n in range(config["range_start"], config["range_end"] + 1):
            algorithm = KMeans(
                n_clusters=n,
                init='k-means++',
                n_init=10,
                max_iter=config["max_iter"],
                tol=config["tol"],
                random_state=111,
                algorithm='elkan'
            )
            algorithm.fit(X)
            inertia.append({"k": n, "value": algorithm.inertia_})
        return inertia
    return []

# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6868)
