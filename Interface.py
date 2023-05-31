import streamlit as st

# this is title of the page
st.set_page_config(
    page_title="Sign Language",
    page_icon=":ok_hand:"#"chart_with_upwards_trend",
    )

st.sidebar.success("Select a demo")

st.title("Welcome dear friend")

st.markdown(
    """
    We can help you interpret [American sign language](https://en.wikipedia.org/wiki/American_Sign_Language) from:
        \n - uploaded image or
        \n - webcam stream
        """
    # Set the font size using HTML tags
    "<h2 style='font-size: 32px;'>👈 Select an option from the dropdown!</h2>   "

    , unsafe_allow_html=True)

# Add an image to the main page
image = "ASL_kaggle_image.png"
st.image(image, use_column_width=True)
