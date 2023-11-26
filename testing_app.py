# Import all of the dependencies
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI

import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
from aiutils import chat_with_ai

from dotenv import load_dotenv, find_dotenv

# Set the layout to the streamlit app as wide 
load_dotenv(find_dotenv())


st.set_page_config(layout='wide')


# Setup the sidebar
with st.sidebar: 
    st.title('Infusing Lip Reading with an LLM')
    st.info("It's important to note that the application's true potential for robustness remains untapped. Given better resources and an extended training duration, the model could evolve significantly. Unfortunately, due to resource constraints, the training period was limited, hindering the model's full development. By expanding the dataset and prolonging the training duration, I firmly believe this Lip Reading application can overcome its current limitations. I couldn't allocate more resources for its training, but I envision a future where this tool evolves into a robust, versatile, and widely applicable solution.")
    st.write("The credit for utils file which contains the data loading functions goes to Nicolas Renotte.") 
    st.write("You can check this out here: https://github.com/nicknochnack/LipNet/blob/main/app/utils.py")
    
    
st.title('LipNet App with Llama') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('data', 'test_videos'))
selected_video = st.selectbox('Select a Video for prediction....', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video  
    with col1: 
        file_path = os.path.join('data', 'test_videos', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test/{selected_video}.mp4 -y')  ##optional if the format is in some other format than 'mp4'.

        # Render the video
        video = open(f'test/{selected_video}.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        st.info("This is the Original Label.")
        st.text(annotations)
        st.info("This is what the Lip Reading Model Predicted.")
        
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        
        st.text("Raw Output:")
        st.text(decoder)

        # Convert prediction to text
        st.text('After decoding, the Output is: ')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
     
    ##AI Infusion   
    template = """
    You are an helpful AI.
    YOur job is to look at the User's input that is given below and repeat what the input was. 
    If there are any incomplete words and correct the input then you will have to complete them and then give yuur final output.
    User : {human_input}
    You answer is: 
    """
        
    prompt = PromptTemplate(template = template, input_variables = ['human_input'])
    
    llm_chain = LLMChain(llm = OpenAI(model = 'Llama-2-13b-chat-hf', temperature = 0.1),
                         prompt = prompt,
                         )
    
    output = llm_chain.run(converted_prediction)
    st.info("LLM's output: \n")
    st.write(output)    
    
    st.info("I invite developers and enthusiasts alike to join in refining this prototype. Together, we can enhance its capabilities, enabling it to decode a broader range of videos and become a reliable tool for various applications.")
    
        
