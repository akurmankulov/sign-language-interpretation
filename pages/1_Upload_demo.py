import streamlit as st

st.set_page_config(
    page_title="SLB Signs",
    page_icon="🌍",
    )

st.sidebar.success("Select another demo")

# creating interface to upload data
# st.markdown(f'# {list(demo_options.keys())[1]}')
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an image file :sunglasses:", type=['png', 'jpg'])

if uploaded_file is not None:
    data = uploaded_file
    st.write("filename:", data.name)
    st.success('An image has been successfully uploaded! You are doing great!')
    st.write("the sign means 'C'")
