import streamlit as st
import requests

# Function to send query to backend and retrieve response
def send_query(question):
    url = 'http://127.0.0.1:5000/query'  # Update if deploying elsewhere
    payload = {'question': question}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get('response')
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Main function to define Streamlit UI
def main():
    st.title("InfoQuest")
    st.markdown(
        """
        <style>
        /* Styling for the background */
        body {
            background-color: offwhite;
        }
        /* Styling for the Send button */
        .css-1yywi0x {
            background-color: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }
        /* Styling for user and bot messages */
        .message-container {
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #e0f7fa; /* Light blue background for user messages */
            padding: 10px;
            border-radius: 10px;
            text-align: right;
            color: black; /* Black text color */
        }
        .bot-message {
            background-color: #e3f2fd; /* Light blue background for bot messages */
            padding: 10px;
            border-radius: 10px;
            text-align: left;
            color: black; /* Black text color */
        }
        .label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state to preserve chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Input field for user question
    question = st.text_input("Type your question here:", key="input_question")

    # Button to submit the question
    if st.button("Send"):
        if question:
            # Send query to backend and get response
            response = send_query(question)

            # Append question and response to history
            st.session_state.history.append((question, "User"))
            st.session_state.history.append((response, "Bot"))

    # Display chat history
    for i, (msg, sender) in enumerate(st.session_state.history):
        if sender == "User":
            st.markdown('<div class="label">You:</div><div class="message-container"><div class="user-message">{}</div></div>'.format(msg), unsafe_allow_html=True)
        elif sender == "Bot":
            st.markdown('<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{}</div></div>'.format(msg), unsafe_allow_html=True)

if __name__ == "__main__":
    main()







# import streamlit as st
# import requests

# # Function to send query to backend and retrieve response
# def send_query(question):
#     url = 'http://127.0.0.1:5000/query'  # Update if deploying elsewhere
#     payload = {'question': question}
#     headers = {'Content-Type': 'application/json'}

#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#             return response.json().get('response')
#         else:
#             return f"Error: {response.status_code}"
#     except requests.exceptions.RequestException as e:
#         return f"Error: {str(e)}"

# # Main function to define Streamlit UI
# def main():
#     st.title("InfoQuest")
#     st.markdown(
#         """
#         <style>
#         /* Styling for the background */
#         body {
#             background-color: powderblue;
#         }
#         /* Styling for the Send button */
#         .css-1yywi0x {
#             background-color: #4CAF50;
#             color: white;
#             border-color: #4CAF50;
#         }
#         /* Styling for user and bot messages */
#         .stTextInput, .stTextArea {
#             color: aqua;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     # Initialize session state to preserve chat history
#     if 'history' not in st.session_state:
#         st.session_state.history = []

#     # Input field for user question
#     question = st.text_input("Type your question here:", key="input_question")

#     # Button to submit the question
#     if st.button("Send"):
#         if question:
#             # Append question to history
#             st.session_state.history.append((question, "User"))

#             # Send query to backend and get response
#             response = send_query(question)

#             # Append response to history
#             st.session_state.history.append((response, "Bot"))

#     # Display chat history
#     for i, (msg, sender) in enumerate(st.session_state.history):
#         if sender == "User":
#             st.text_input("You:", value=msg, disabled=True, key=f"User_{i}")
#         elif sender == "Bot":
#             st.text_area("Bot:", value=msg, disabled=True, key=f"Bot_{i}")

# if __name__ == "__main__":
#     main()