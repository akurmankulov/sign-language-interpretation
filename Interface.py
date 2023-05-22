import streamlit as st

# setting up a title
'Sign language project welcomes you'

# creating interface to upload data
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an image file :sunglasses:", type=['png', 'jpg'])

if uploaded_file is not None:
    data = uploaded_file
    st.write("filename:", data.name)