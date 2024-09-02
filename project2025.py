from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import os 
import streamlit as st


def load_embeddings():
    embedding_model = "sentence-transformers/all-MiniLM-L6-V2"
    embeddings = HuggingFaceEmbeddings(
        model_name = embedding_model
    )
    return embeddings


def load_faiss_index(embeddings):

    # load the FAISS index
    db = FAISS.load_local("faiss_db/", embeddings, allow_dangerous_deserialization=True)
    
    return db 


def get_llm_model():
    groq_api_key = st.secrets['groq_api_key']
    os.environ['GROQ_API_KEY'] = groq_api_key

    model_name = 'llama-3.1-8b-instant'

    model = ChatGroq(
        model=model_name
    )

    return model


def get_prompt(query, vector_index):

    results = vector_index.similarity_search_with_score(query=query, k=3)

    context = results[0][0].page_content + '\n' + results[1][0].page_content 

    prompt = f"""
    You are a Helpful Chat Assistant for Project 2025
    You have to Answer Queries only based on the Context Provided.

    Context: {context} 

    Instructions: 
    1. Stick to Answering the Questions, Dont Say anything Like >> According to the provided context, etc.
    2. Always Try to Give Best Answer from Context.
    3. Do not Provide Any Extra Info Other than Context. You can rewrite in your own words without changing context.
    4. Never Respond with Something Like -> 'context provided does not explicitly mention', etc.
    5. If you cant answer from Context Politely tell user to ask questions related to Project 2025 only. 
    6. Do not Reveal you Internal Working. eg. never mention 'context'.

    Question: {query}

    Answer:"""

    return prompt


def generate_response(prompt, model):
    response = model.invoke(prompt)
    return response.content 


def main():
    
    # Title of the Streamlit app
    st.title(":newspaper: Project 2025 RAG Chatbot")

    # Load Embeddings
    embeddings = load_embeddings()

    # Get FAISS Index
    faiss_index = load_faiss_index(embeddings)

    # Get LLM Model
    model = get_llm_model()

    # Initialize chat history in the session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        # Reset the chat history to an empty list
        st.session_state['chat_history'] = []
        # Refresh the app to reflect the cleared chat history
        # st.rerun()

    # Display the chat history
    for i, message in enumerate(st.session_state.chat_history):
        # Use Streamlit's chat message function to render each message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input box for users to type their queries
    query = st.chat_input("Ask a question about the Project 2025:")

    if query:
        # Add the user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Display the user question in the chat interface
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):

                # Get Prompt
                prompt = get_prompt(query, faiss_index)

                # Generate Response
                response = generate_response(prompt, model)

                # Display the generated response
                st.write(response)    

                # Add the assistant's response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})



# Run the main function if this script is executed
if __name__ == '__main__':
    main()
