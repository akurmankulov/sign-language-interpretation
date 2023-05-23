import streamlit as st

picture = st.camera_input("Take a pic!")

if picture:
    st.image(picture)
