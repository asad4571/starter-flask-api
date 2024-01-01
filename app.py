# app.py
from flask import Flask, render_template, request
import nltk
nltk.download('stopwords')
from sentence_transformers import SentenceTransformer, util  # Import SentenceTransformer

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

app = Flask(__name__)


import os
from werkzeug.utils import secure_filename

def extract_text_from_pdf(file):
    # Create a temporary directory if it doesn't exist
    temp_dir = 'temp_pdfs'
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file to the temporary directory
    filename = secure_filename(file.filename)
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)

    # Extract text from the PDF
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Remove the temporary file
    os.remove(file_path)

    return text



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rank', methods=['POST'])
def rank_resumes():
    # Get job description and resume files from the form
    job_description = request.form['job_description']
    resume_files = request.files.getlist('resumes')

    # Create a list to store resume texts and filenames
    resume_texts = []
    resume_filenames = []

    # Preprocess the text data
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Get embeddings for the job description
    job_embedding = model.encode(job_description)

    # Get embeddings for each resume
    resume_embeddings = []
    for resume_file in resume_files:
        resume_text = extract_text_from_pdf(resume_file)
        resume_texts.append(resume_text)
        resume_embedding = model.encode(resume_text)
        resume_embeddings.append(resume_embedding)
        resume_filenames.append(secure_filename(resume_file.filename))

    # Calculate cosine similarity between job description and each resume
    similarities = [util.pytorch_cos_sim(job_embedding, resume_embedding)[0][0].tolist() for resume_embedding in resume_embeddings]

    # Create a list of tuples (resume text, similarity, resume filename) for ranking
    ranked_resumes = list(zip(resume_texts, similarities, resume_filenames))
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)

    return render_template('result.html', ranked_resumes=ranked_resumes)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
