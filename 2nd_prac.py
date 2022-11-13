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
