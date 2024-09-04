import streamlit as st
from PIL import Image

st.markdown(
    """
    <style>
    .full-height {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
    }

    .top-gap {
        margin-top: 32px;
    }

    .sub-text {
        color: gray; 
        text-align: center;
        width: 100%;
    }

    .centered-content {
        display: flex;
        width: "100%";
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .small-header {
        font-size: 16px;
        color: #FF4B4B;
        margin-top: 128px;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------- HERO


col1, col2 = st.columns(2, gap="large", vertical_alignment="center")

with col1:
    st.title('AEGIS - The Autonomous Embedded-Graph Intrusion Sentinel')
    st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer non elit quis augue blandit suscipit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.')

    st.markdown('<div class="top-gap"></div>', unsafe_allow_html=True)

    # Create a horizontal container for the buttons
    button_container = st.container()

    # Add buttons in separate columns within the container
    button_col1, button_col2, _ = button_container.columns([1, 1, 1])

    with button_col1:
        if st.button('Evaluate Agent', use_container_width=True, type="primary"):
            st.switch_page("evaluation.py")

    with button_col2:
        if st.button('Train Agent', use_container_width=True, type="secondary"):
            st.switch_page("training.py")

# Column 2: Hero graphic
with col2:
    st.image(use_column_width=True, image='../tower_defense.png')

st.markdown('<p class="sub-text top-gap">read more ↓</p>', unsafe_allow_html=True)


# --------------------------------------- DOCUMENTATION

st.markdown(
    f"""
    <div class="centered-content margin-bottom">
        <p class="small-header">DOCUMENTATION</p>            
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------- DELIVERABLES

st.markdown(
    f"""
    <div class="centered-content margin-bottom">
        <p class="small-header">DELIVERABLES</p>            
    </div>
    """,
    unsafe_allow_html=True
)
