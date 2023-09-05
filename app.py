import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import Replicate
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

template = '''You analyze the sentiment of the text provided. 
The sentiment can be positive, negative, or neutral.Do not guess if you 
don't know.
{text}
'''
prompt = PromptTemplate(template=template, input_variables=["text"])

#initialize Replicate
llm = Replicate(
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    input={"temperature": 0.01, "max_length": 500, "top_p": 1}
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Define the Streamlit app
def main():
    # Set the title and description
    st.title("Sentiment Analyzer ChatBot :book:")
    st.write("Enter your text, and this app will provide an answer.")

    # User input
    user_input = st.text_area("Enter your text here:")

    if st.button("Send"):
        # Analyze the user's input
        result = llm_chain.run(user_input)
        st.write("Answer:")
        st.write(result)

if __name__ == "__main__":
    main()

