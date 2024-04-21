import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from PIL import Image  # Import Image module from PIL

import numpy as np

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('model-ai.h5')

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    # st.write("Reply: ", response["output_text"])
    return response["output_text"]

@app.route('/chatPDF', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('message')

    pdf_path = os.path.abspath("ideal catalogue.pdf")

    
    if pdf_path:
        if not os.path.exists(pdf_path):
            return jsonify({"status": "error", "message": f"File not found: {pdf_path}"}), 404

        raw_text = get_pdf_text([pdf_path])
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    # Get the response
    response_text = user_input(user_question)

    return jsonify({"response": response_text})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_question = data.get('message')

    pdf_path = os.path.abspath("training.pdf")

    
    if pdf_path:
        if not os.path.exists(pdf_path):
            return jsonify({"status": "error", "message": f"File not found: {pdf_path}"}), 404

        raw_text = get_pdf_text([pdf_path])
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    # Get the response
    response_text = user_input(user_question)

    return jsonify({"response": response_text})


def preprocess_user_photo(img_file):
    # Load image using PIL (Python Imaging Library)
    img = Image.open(img_file)

    # Resize image to the required dimensions (160, 160) using PIL
    img = img.resize((160, 160))

    # Convert image to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Expand dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)

    # Rescale pixel values to the range [0, 1]
    img_array /= 255.0

    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        image_file = request.files['file']
        
        # preprocess the image
        preprocessed_img = preprocess_user_photo(image_file)

        # Make predictions using the loaded model
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions)
        classes = ['dress','pants','shirt','shoes','shorts','dress','pants','shirt','shoes','shorts','pants','shoes','shorts','pants','shoes','shorts','dress','pants','shoes','dress','pants','shoes','shorts']
        predicted_class = classes[predicted_class_index]
        print(predicted_class)
        
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)