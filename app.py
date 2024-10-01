import streamlit as st


# --- PAGE SETUP ---
about_page = st.Page(
    "views/introduction.py",
    title="About",
    icon="🅰",
    default=True,
)
project_1_page = st.Page(
    "views/krishnaji_chat.py",
    title="Chai With Krishnaji",
    icon="🕉",
)


# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Projects": [project_1_page],
    }
)


# --- SHARED ON ALL PAGES ---
#st.logo("assets/codingisfun_logo.png")
st.sidebar.markdown("Connect with [Aman](https://www.linkedin.com/in/aman-naik/)")


# --- RUN NAVIGATION ---
pg.run()