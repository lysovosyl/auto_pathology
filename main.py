import streamlit as st
from multipage import MultiPage
from Pages import home,detection,training,patches,labeling


st.set_page_config(page_title="H&E Processor",page_icon=":trigger:",layout="wide")
st.title("H&E Processor")

app=MultiPage()

app.add_page("Home",home.app)
app.add_page("H&E to Patches",patches.app)
app.add_page("Labeling",labeling.app)
app.add_page("Training",training.app)
app.add_page("Detection",detection.app)





if __name__=="__main__":
    app.run()

