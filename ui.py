import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from joblib import Parallel, delayed
import joblib
st.title("PARKINSON'S DISEASE PREDICTION")
st.set_option('deprecation.showPyplotGlobalUse', False)
data = pd.read_csv("./Parkinsson disease.csv")
data.drop('name',axis=1,inplace=True)
nav = st.sidebar.radio("NAVIGATION", ["HOME", "PREDICT"])
X=data[['MDVP:Fo(Hz)','MDVP:Flo(Hz)','MDVP:Shimmer','MDVP:APQ','spread1','spread2','PPE']]
model=joblib.load(open('rfc_model.pkl', 'rb'))
if nav == "HOME":
    st.image("park.png", 50, 50, True)
    st.markdown(""" ### INFORMATION ABOUT THE DATASET""")
    if st.checkbox("DETAILED DESCRIPTION"):
        st.markdown("""
        ### MDVP:F0 (Hz)	Average vocal fundamental frequency
        ### MDVP:Fhi (Hz)	Maximum vocal fundamental frequency
        ### MDVP:Flo (Hz)	Minimum vocal fundamental frequency
        ### MDVP:Jitter(%)	MDVP jitter in percentage
        ### MDVP:Jitter(Abs)	MDVP absolute jitter in ms
        ### MDVP:RAP	MDVP relative amplitude perturbation
        ### MDVP:PPQ	MDVP five-point period perturbation quotient
        ### Jitter:DDP	Average absolute difference of differences between jitter cycles
        ### MDVP:Shimmer	MDVP local shimmer
        ### MDVP:Shimmer(dB)	MDVP local shimmer in dB
        ### Shimmer:APQ3	Three-point amplitude perturbation quotient
        ### Shimmer:APQ5	Five-point amplitude perturbation quotient
        ### MDVP:APQ11	MDVP 11-point amplitude perturbation quotient
        ### Shimmer:DDA	Average absolute differences between the amplitudes of consecutive periods
        ### NHR	Noise-to-harmonics ratio
        ### HNR	Harmonics-to-noise ratio
        ### RPDE	Recurrence period density entropy measure
        ### D2	Correlation dimension
        ### DFA	Signal fractal scaling exponent of detrended fluctuation analysis
        ### Spread1	Two nonlinear measures of fundamental
        ### Spread2	Frequency variation
        ### PPE	Pitch period entropy
        """)
    st.markdown("""### DATASET: PARKINSON DISEASE.CSV""")
    st.dataframe(data)
    st.markdown(""" #### SHAPE OF THE DATASET:(195,24) """)
    if st.checkbox("SHOW TABLE"):
        st.table(data)
    st.markdown(""" ## EXPLORATORY DATA ANALYSIS""")
    if st.checkbox("FEATURES WITH MAXIMUM CORRELATION WITH TARGET CLASS"):

        fig_=plt.figure(figsize=(10, 5))
        sns.countplot(x=data['status'])
        plt.xlabel("NUMBER OF DISTINCT TARGET CLASS LABELS")
        plt.ylabel("COUNT")
        st.pyplot(fig_)

        fig=plt.figure(figsize=(10, 5))
        plt.title("PPE VS SPREAD-1")
        sns.regplot(x=data['PPE'], y=data['spread1'])
        plt.xlabel("PITCH PERIOD ENTROPY")
        plt.ylabel("SPREAD-1")
        st.pyplot(fig=fig)

        fig_=plt.figure(figsize=(10, 5))
        plt.title("MDVP:JITTER VS MDVP:RAP")
        sns.regplot(x=data['MDVP:Jitter(Abs)'],y=data['MDVP:RAP'])
        plt.xlabel("'MDVP:Jitter(Abs)")
        plt.ylabel("MDVP:RAP")
        st.pyplot(fig_)
    elif st.checkbox("CLICK HERE TO SEE THE RELATION AMONG ALL COLUMNS"):
        x_vals=st.selectbox("SELECT X-AXIS COLUMNS",options=X.columns)
        y_vals=st.selectbox("SELECT Y-AXIS COLUMNS",options=X.columns)
        plot=px.scatter(data,x=x_vals,y=y_vals)
        plot.update_layout(
        font=dict(
        family="Calibri",
        size=20,  # Set the font size here
        color="green"
    )
)
        col1, col2, col3 = st.columns(3)
        if col2.button("PLOT"):
            st.plotly_chart(plot)

        # st.write("INTERACTIVE PLOT")
        # layout=gb.Layout(
        #     xaxis=dict(range=[0,2000]),
        #     yaxis=dict(range=[0,2000])
        # )
        # fig=gb.Figure(data=gb.Bar(x=data['Pregnancies'],y=data['Insulin'],mode='marker'),layout=layout)
        # st.plotly_chart(fig)
# 0.121009,1.737287,-0.860817,-0.944288,-0.757011,-0.823129,-0.758083,-0.971711,
# -0.924538,-0.993521,-0.888872,-0.876144,-0.993519,-0.589551,2.181916,-1.431704,
# 0.431668,-1.870140,-0.649697,0.137057,-1.616379

# -0.994279,-0.758967,-0.152844,0.764967,1.232424,0.611243,1.425912,0.612212,1.058449,0.885794
# ,1.036706,1.570486,0.576632,1.036405,-0.324961,-0.209630,-0.629027,1.974581,
# 1.087645,1.009449,-0.127555,1.333395
elif nav == "PREDICT":
    st.markdown("""EXAMPLE DATA [95.056000,91.226000,0.028380,0.024440,-5.011879,0.325996,0.271362] HAD THE DISEASE """)
    st.markdown("""EXAMPLE DATA [202.266000,197.07900,00.009540,0.007190,-7.695734,0.178540,0.056141] DOESN'T HAD DISEASE """)
    mdvp_1=st.selectbox("ENTER MDVP:F0 ",(95.056000,202.266000,122.964,209.144))
    mdvp_2=st.selectbox("ENTER MDVP:Flo (Hz)",(91.226000,197.07900,114.676,109.379))
    mdvp_3=st.selectbox("ENTER MDVP:Shimmer",(0.028380,0.009540,0.01681,0.01861))
    mdvp_4=st.selectbox("ENTER MDVP:APQ",(0.024440,0.007190,0.01400,0.01382))
    spread_1=st.selectbox("ENTER SPREAD_1",(-5.011879,-7.695734,-6.482096,-7.040508))
    spread_2=st.selectbox("ENTER SPREAD_2",(0.325996,0.178540,0.264967,0.066994))
    ppe=st.selectbox("ENTER PPE",(0.271362,0.056141,0.128872,0.101516))
    inputs=[mdvp_1,mdvp_2,mdvp_3,mdvp_4,spread_1,spread_2,ppe]
    inputs=np.array(inputs).reshape(1,-1)
    pred=model.predict(inputs)
    col1, col2, col3 = st.columns(3)
    if col2.button("PREDICT"):
        if pred==0:
            st.balloons()
            st.write("YOU DON'T HAVE DISEASE")
        elif pred==1:
            st.write("YOU HAVE THE DISEASE")
    # if st.checkbox("SUPPORT VECTOR MACHINE"):
    #     st.markdown(model.score(inputs,data[[inputs[0]==data['MDVP:Fo(Hz)']]]['status'])) 
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)