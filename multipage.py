import streamlit as st
class MultiPage:
    def __init__(self):
        self.page = []

    def add_page(self,title,func):
        self.page.append(
            {
                "title":title,
                "function":func
            }
        )

    def run(self):
        st.sidebar.title('Navigation')
        page=st.sidebar.selectbox(
            "Function",
            self.page,
            format_func=lambda page: page["title"]
        )
        page["function"]()
