import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

model = pickle.load(open('st.pkl', 'rb'))

df=pd.read_csv("df1.csv")
# print(df.shape)
# print(df.shape)

X=df.drop(columns=["rent","activation_date"])
Y=df.rent

# print(X.shape)

std=RobustScaler()

X_processed=std.fit_transform(X)


st.markdown(
    """
    <style>
    .title {
    margin-top:0px;
    color: #FF5733; 
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
}

.text {
    color: #EFA18A; 
    font-size: 20px;
    font-weight: italic;
    text-align: center;
    margin-bottom: 20px;
    font-wright:500
}
.prediction {
    color: #FF5733; 
    font-size: 20px;
    font-weight: italic;
    text-align: center;
    margin-bottom: 20px;
    font-wright:500
}
.container {
        padding: 20px;
        background-color: #f5f5f5;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
.form {
        margin-bottom: 20px;
    }
.form-header {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
}
.form-input {
    margin-bottom: 10px;
}
.form-button {
    background-color: #4CAF50;
    color: white;
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    padding: 0.375rem 0.75rem;
    font-size: 1rem;
    border-radius: 0.25rem;
    border: none;
}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Predict House Rent</div>', unsafe_allow_html=True)


st.markdown('<div class="text">Enter Property Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    housetype=st.selectbox('House Type', ['BHK1','BHK2','BHK3','BHK4','BHK4PLUS','RK1'],help='Pick a House Type')
    latitude = st.number_input("Latitude",value= 0.0,step=0.01, help='Enter the latitude')
    longitude = st.number_input("Longitude",value= 0.0,step=0.01, help='Enter the longitude')
    leasetype=st.selectbox('Lease Type', ['FAMILY','ANYONE','BACHELOR','COMPANY'],help='Pick a Lease Type')
    gym = st.selectbox("Gym", ['Yes', 'No'], key='gym')
    lift = st.selectbox("Lift", ['Yes', 'No'], key='lift')

with col2:
    swimming_pool = st.selectbox("Swimming Pool", ['Yes', 'No'], key='swimming_pool')
    negotiable = st.selectbox("Negotiable", ['Yes', 'No'], key='negotiable')
    furnishing=st.selectbox('Furnishing Type', ['Semi-Furnished','Not-Furnished','Fully-Furnished'],help='Pick a Furnishing Type')
    parking=st.selectbox('Parikng Type', ['Four_Wheeler','Two_Wheeler','None'],help='Pick a Parking Type')
    property_size = st.number_input("Property Size", 0,1000, help='Enter the property size')
    property_age = st.number_input("Property Age", 0,1000, help='Enter the property age')
    

with col3:
    bathroom = st.text_input("Bathroom", 0,1000, help='Enter the number of bathrooms')
    facing=st.selectbox('Direction', ['E','N','W','S','NE','SE','NW','SW'],help='Pick a Direction')
    cupboard = st.number_input("Cupboard", 0,1000, help='Enter the number of cupboards')
    floor = st.number_input("Floor", 0,1000, help='Enter the number of floor')
    totalfloor = st.number_input("Total Floor", 0,1000, help='Enter the total number of floor')
    internet = st.selectbox("Internet", ['Yes', 'No'], key='Internet')

col4, col5, col6 = st.columns(3)
with col4:
    ac = st.selectbox("AC", ['Yes', 'No'], key='ac')
    club = st.selectbox("Club", ['Yes', 'No'], key='club')
    intercom = st.selectbox("Intercom", ['Yes', 'No'], key='intercom')
    pool = st.selectbox("Pool", ['Yes', 'No'], key='pool')
    fs = st.selectbox("FS", ['Yes', 'No'], key='fs')
    cpa = st.selectbox("CPA", ['Yes', 'No'], key='cpa')

with col5:
    servant = st.selectbox("Servant", ['Yes', 'No'], key='servant')
    security = st.selectbox("Security", ['Yes', 'No'], key='security2')
    sc = st.selectbox("SC", ['Yes', 'No'], key='sc')
    gp = st.selectbox("GP", ['Yes', 'No'], key='gp')
    park = st.selectbox("Park", ['Yes', 'No'], key='park')
    rwh = st.selectbox("RWH", ['Yes', 'No'], key='rwh')
    balconies = st.number_input("Balconies", 0,1000, help='Enter the number of Balconies')
    
with col6:
    stp = st.selectbox("STP", ['Yes', 'No'], key='stp')
    hk = st.selectbox("HK", ['Yes', 'No'], key='hk')
    pb = st.selectbox("PB", ['Yes', 'No'], key='pb')
    vp = st.selectbox("VP", ['Yes', 'No'], key='vp')
    watersupply=st.selectbox('Water Supply Type', ['CORP_BORE','CORPPORATION','BOREWELL'],help='Pick a Water Supply Type')
    buildingtype=st.selectbox('Building Type', ['IF','AP','IH','GC'],help='Pick a Building Type')
    
button_style = """
    <style>
    .stButton > button:first-child {
        background-color: #FF5733;
        color: white;
        padding: 0.375rem 0.75rem;
        font-size: 1rem;
        border-radius: 0.25rem;
        border: none;
        margin-left:300px
    }
    </style>
    """
        
st.markdown(button_style, unsafe_allow_html=True)
predict_button = st.button("Predict")

if predict_button:
    input_values=[]
    input_values.append(latitude)
    input_values.append(longitude)
    input_values.append(gym)
    input_values.append(lift)
    input_values.append(swimming_pool)
    input_values.append(negotiable)
    input_values.append(property_size)
    input_values.append(property_age)
    input_values.append(bathroom)
    input_values.append(cupboard)
    input_values.append(floor)
    input_values.append(totalfloor)
    input_values.append(balconies)
    input_values.append(housetype)
    input_values.append(leasetype)
    input_values.append(furnishing)
    input_values.append(parking)
    input_values.append(facing)
    input_values.append(watersupply)
    input_values.append(buildingtype)
    input_values.append(internet)
    input_values.append(ac)
    input_values.append(club)
    input_values.append(intercom)
    input_values.append(pool)
    input_values.append(cpa)
    input_values.append(fs)
    input_values.append(servant)
    input_values.append(security)
    input_values.append(sc)
    input_values.append(gp)
    input_values.append(park)
    input_values.append(rwh)
    input_values.append(stp)
    input_values.append(hk)
    input_values.append(pb)
    input_values.append(vp)
    m1={"BHK1":[1,0,0,0,0,0],"BHK2":[0,1,0,0,0,0],"BHK3":[0,0,1,0,0,0],"BHK4":[0,0,0,1,0,0],"BHK4PLUS":[0,0,0,0,1,0],"RKPLUS":[0,0,0,0,0,1],
    "ANYONE":[1,0,0,0],"BACHELOR":[0,1,0,0],"COMPANY":[0,0,1,0],"FAMILY":[0,0,0,1],
    "Fully-Furnished":[1,0,0],"Not-Furnished":[0,1,0],"Semi-Furnished":[0,0,1],
    "Both":[1,0,0,0],"Four_Wheeler":[0,1,0,0],"Two_Wheeler":[0,0,1,0],"None":[0,0,0,1],
    "E":[1,0,0,0,0,0,0,0],"N":[0,1,0,0,0,0,0,0],"NE":[0,0,1,0,0,0,0,0],"NW":[0,0,0,1,0,0,0,0],"NW":[0,0,0,0,1,0,0,0],"S":[0,0,0,0,0,1,0,0],"SW":[0,0,0,0,0,0,1,0],"W":[0,0,0,0,0,0,0,1],
    "BOREWELL":[1,0,0],"CORPPORATION":[0,1,0],"CORP_BORE":[0,0,1],
    "AP":[1,0,0,0],"IF":[0,1,0,0],"IH":[0,0,1,0],"GC":[0,0,0,1],"Yes":[1],"No":[0],"0":[0]}
    f=[]
    # print(input_values)
    for i in input_values:
        if i in m1:
            l=m1[i]
            for j in l:
                f.append(float(j))
        else:
            f.append(float(i))
    # print(len(f))
    f=np.array(f)
    feat=std.transform(f.reshape(1,-1))
    # print(feat)
    pred=model.predict(feat)
    pred_ans=pred[0]
    pred_ans=round(pred_ans,0)
    pred_ans=int(pred_ans)  
    rupee_symbol = '\u20B9'
    prediction = f"Your predicted rent:{rupee_symbol}{pred_ans}"
    
    st.markdown('<div class="prediction">{}</div>'.format(prediction), unsafe_allow_html=True)
