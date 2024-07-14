Document Q&A Buddy 🪄

Overview
Document Q&A Buddy is a sophisticated application that leverages advanced Generative AI models, including Google's Gemini and Groq's open-source model, to provide accurate answers to questions based on document content. This project is designed to demonstrate the power of AI in document processing and information retrieval.

Features
Document Ingestion: Efficiently load and process PDF documents.
Advanced Embeddings: Utilize Google's Gemini for high-quality vector embeddings.
Efficient Retrieval: Implement FAISS for fast and accurate vector search.
Interactive Q&A: Seamlessly ask questions and get context-aware answers.
User-Friendly Interface: Built with Streamlit for an intuitive user experience.
Installation
Prerequisites
Python 3.8 or higher
pip (Python package installer)

Clone the Repository : 

git clone https://github.com/yourusername/document-qa-buddy.git
cd document-qa-buddy


Set Up a Virtual Environment : 

python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`


pip install -r requirements.txt


Configure Environment Variables
Create a .env file in the project root directory and add your API keys:

groq_api=YOUR_GROQ_API_KEY
google_api_key=YOUR_GOOGLE_API_KEY

Usage
Running the Application


streamlit run app.py


Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

Credits
Inspired by Krishna Naik Sir.

Author
Made by Sanskar Khandelwal with ❤️‍🔥

License
This project is licensed under the MIT License.



