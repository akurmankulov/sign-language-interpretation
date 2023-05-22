import streamlit as st

st.set_page_config(
    page_title="SLB Signs",
    page_icon="👋",
    )

st.sidebar.success("Select another demo")

st.markdown("Work in progress...")

#read image from webcam
picture = st.camera_input("Let's get your webcam in action and grab a picture...")
if picture:
    st.image(filters_to_funcs[filters](picture), channels="BGR")