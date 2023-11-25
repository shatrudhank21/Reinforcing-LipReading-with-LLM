from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI


def chat_with_ai(human_input):
    template = f"""
    You are an helpful AI. You will provide a brief context of what the user tells you.
    You will be provided with a little text over line. The text you will get is the result of a Lip reading application.
    On the basis of the text, you will try to figure out what the User is trying to say.
    Do not make up wrong answers, always answer only when you are very confident about it.
    You are an AI created for the purpose of taking care of the lacking output from the Lip reading application.
    You will have to figure out the what the user is trying to say and predict the output.
    
    Human Input is given as follows:
    User : {human_input}
    """
        
    prompt = PromptTemplate(template = template, input_variables = ['human_input'])
    
    llm_chain = LLMChain(llm = OpenAI(model = 'gpt-3.5-turbo-0301', temperature = 0.1),
                         prompt = prompt,
                         )
    
    output = llm_chain.run(human_input)
    
    return output    
