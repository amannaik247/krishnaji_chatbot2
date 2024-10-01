
import streamlit as st

# from forms.contact import contact_form


#@st.experimental_dialog("Contact Me")
#def show_contact_form():
#    contact_form()


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    st.image("./assets/Comming soon-01-01.png", width=200)

with col2:
    st.title("Aman Naik", anchor=False)
    st.write(
        "Pursuing AI/ML. Love doing 3D art as a hobby!"
    )
    #if st.button("✉️ Contact Me"):
        #show_contact_form()


# --- EXPERIENCE & QUALIFICATIONS ---
st.write("\n")
st.subheader("About Chai with Lord Krishna", anchor=False)
st.write(
    """
    - Lord Krishna is a renowned Hindu spiritual leader and author of the Bhagavad Gita.
    - Chai is a popular Indian drink made from roasted and ground black pepper.
    - The combination of Chai and Lord Krishna's spiritual wisdom has been used in various cultures for centuries.
    - This app aims to bring the blend of Chai and Lord Krishna's wisdom to the world through AI/ML.
    """
)

# --- SKILLS ---

st.write("\n")
st.subheader("Expected additions", anchor=False)
st.write(
    """
    - AI/ML model training
    - User authentication and authorization
    - Interactive chat interface
    - More features and enhancements
    """
)
