import streamlit as st

st.set_page_config(
    page_title="SLB Signs",
    page_icon=":selfie:"#"👋",
    )

st.sidebar.success("Select another demo")

st.title("Shall we try to do it live?!")

#read image from webcam
picture = st.camera_input("Let's get your webcam in action and grab a picture...")
if picture:
    st.write("The letter is K")