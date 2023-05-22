import streamlit as st

# setting up a title
st.set_page_config(
    page_title="SLB Signs",
    page_icon="chart_with_upwards_trend",
    )

st.sidebar.success("Select a demo")

st.markdown(
    """https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
    **👈 Select an option from the dropdown!

    We can help you interpret American sign language from:
        1. uploaded image
        2. webcam stream
            
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