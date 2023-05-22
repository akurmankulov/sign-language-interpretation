import streamlit as st

# this is title of the page
st.set_page_config(
    page_title="SLB Signs",
    page_icon="chart_with_upwards_trend",
    )

st.sidebar.success("Select a demo")

st.markdown(
    """
    **👈 Select an option from the dropdown!

    We can help you interpret American sign language from:
        \n 1. uploaded image
        \n 2. webcam stream
            
    """
)

# demo_options = {
#     "—": intro,
#     "Prediction from uploaded image": upload_demo,
#     "Preditopn from webcam": webcam_demo,
#     "References": team_info
# }

# demo_name = st.sidebar.selectbox("Choose an option", demo_options.keys())
# demo_options[demo_name]()