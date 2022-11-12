import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# make containers
header = st.container()              # here i forgto to write (). so i was gettin na error
data = st.container()
features = st.container()
model_train = st.container()

with header:
    st.title("penguine ki app")
    st.text("in this project we will work penguine data")
    
with data:
    # loading and dealing with data
    st.header("penguine nai udta.")
    df = sns.load_dataset("penguins")
    st.write(df.head(10))
    st.write(df.describe())
    st.write(df.isnull().sum())
    st.write(df.shape)
    df =df.dropna()
    
    # plotting graphs in streamlit
    st.subheader("penguine ky gender ky lehaz sy farak")
    st.bar_chart(df["sex"].value_counts())
    
    # again plotting graphs in streamlit
    st.subheader("bill_depth ky lehaz sy farak")    
    st.bar_chart(df['bill_depth_mm'].sample(10))
            

with features:
    st.header("this is our data features")
    st.markdown("1- **features**: This will tell about penguines")
    
with model_train:
    st.header("penguine ka kia scene hy")
    #making couloumsn.
    input,display= st.columns(2)
    
    # first couloumns.
    max_depth= input.slider("how much depth do you want", min_value=0, max_value= 20, value= 4, step=2)    

n_estimators = input.selectbox("how much body mass should be: ", options=[3750,3250, 3625, 3800])

input.write(df.columns)

input_feat = input.text_input("which feature we should use and the feature must have a data type float or int. ")

# machine learning ka model lgana hy
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
x = df[[input_feat]]
y= df[['flipper_length_mm']]
model.fit(x,y)
pred= model.predict(y)


# model working, by displaying metrices
display.subheader("mean abolute error of the model is: ")
display.write(mean_absolute_error(y, pred))
display.subheader("mean squared error of the model is: ")
display.write(mean_squared_error(y, pred))
display.subheader("R square score of the model is: ")
display.write(r2_score(y, pred))



st.header("Animation wlay graph")



# importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import plotly.express as px

# importing datasets

st.title("plotly ky graph streamlit mein display krwany")
df = px.data.wind()
st.write(df.sample(10))
st.write(df.columns)

# summry stat
st.write(df.describe())
st.write(df.shape)
st.write(df['strength'].max())
st.write(df['frequency'].max())
st.write(df['frequency'].min())

# data management

dist = df["direction"].unique().tolist()
district = st.selectbox("jaldi naa direction winner chose kr", dist)
# df = df[df["direction"]==dist]   

# # plotting

graph_op = px.scatter(df, x= "frequency", y= "strength", color="direction", 
                      range_y=[0,7], range_x=[0.05, 2.6],
                      
                      animation_frame="strength", animation_group="strength")
st.write(graph_op) 

# graph-2

graph_op2 = px.scatter(df, x= "frequency", y= "strength", color="direction", 
                      range_y=[0,4], range_x=[0.05, 2.6],
                      
                      animation_frame="frequency", animation_group="strength")
st.write(graph_op2) 

# with no animation

graph_op3 = px.scatter(df, x= "frequency", y= "strength", color="direction", 
                      range_y=[0,4], range_x=[0.05, 2.6],)
                      
                    #   animation_frame="frequency", animation_group="strength")
st.write(graph_op3) 
