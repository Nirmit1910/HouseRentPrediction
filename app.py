import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import RobustScaler
import re


TEMPLATES_AUTO_RELOAD=True

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

df=pd.read_csv("df1.csv")
print(df.shape)
print(df.shape)

X=df.drop(columns=["rent","activation_date"])
Y=df.rent

print(X.shape)

std=RobustScaler()

X_processed=std.fit_transform(X)

m1={"BHK1":[1,0,0,0,0,0],"BHK2":[0,1,0,0,0,0],"BHK3":[0,0,1,0,0,0],"BHK4":[0,0,0,1,0,0],"BHK4PLUS":[0,0,0,0,1,0],"RKPLUS":[0,0,0,0,0,1],
"ANYONE":[1,0,0,0],"BACHELOR":[0,1,0,0],"COMPANY":[0,0,1,0],"FAMILY":[0,0,0,1],
"Fully-Furnished":[1,0,0],"Not-Furnished":[0,1,0],"Semi-Furnished":[0,0,1],
"Both":[1,0,0,0],"Four_Wheeler":[0,1,0,0],"Two_Wheeler":[0,0,1,0],"None":[0,0,0,1],
"E":[1,0,0,0,0,0,0,0],"N":[0,1,0,0,0,0,0,0],"NE":[0,0,1,0,0,0,0,0],"NW":[0,0,0,1,0,0,0,0],"NW":[0,0,0,0,1,0,0,0],"S":[0,0,0,0,0,1,0,0],"SW":[0,0,0,0,0,0,1,0],"W":[0,0,0,0,0,0,0,1],
"BOREWELL":[1,0,0],"CORPPORATION":[0,1,0],"CORP_BORE":[0,0,1],
"AP":[1,0,0,0],"IF":[0,1,0,0],"IH":[0,0,1,0],"GC":[0,0,0,1]}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTMLGUI
    '''
    float_features = [str(x) for x in request.form.values()]
    # print(len(float_features))
    f=[]
    # print(float_features)
    for i in float_features:
        print(i+" ")
        if i.isdigit()==True:
            f.append(float(i))
        else:
            l=m1[i]
            for j in l:
                f.append(float(j))
    print(len(f))
    f=np.array(f)
    feat=std.transform(f.reshape(1,-1))
    print(feat)
    pred=model.predict(feat)
    pred_ans=pred[0]
    pred_ans=round(pred_ans,0)
    pred_ans=int(pred_ans)
    return render_template('index.html', prediction_text='House Rent is  {}'.format(pred_ans))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

