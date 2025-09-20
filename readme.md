Resume Checker - Hackathon Project
📌 Overview
Resume Checker is a Streamlit-based web application designed to analyze resumes against job descriptions. It helps recruiters and applicants quickly evaluate how well a resume matches a given job description. The system leverages NLP techniques to extract key skills, compare them with job requirements, and generate compatibility results.
________________________________________
🚀 Features
•	Upload multiple resumes (PDF) and job descriptions (PDF).
•	Extracts and analyzes skills, keywords, and experience.
•	Compares resume content with job requirements.
•	Stores results in a SQLite database.
•	Provides a user-friendly Streamlit interface.
•	Includes sample resumes and job descriptions for testing.
________________________________________
📂 Project Structure
Resume Chacker-Hackathon/
│── app.py                   # Main Streamlit application
│── requirements.txt         # Python dependencies
│── resume_results.db        # SQLite database for storing results
│── .streamlit/secrets.toml  # Streamlit configuration (e.g., database/API keys)
│
├── Sample Data/
│   ├── Job Description/
│   │   ├── sample_jd_1.pdf
│   │   └── sample_jd_2.pdf
│   └── Resume/
│       ├── resume - 1.pdf
│       ├── Resume - 2.pdf
│       ├── ...
│       └── Resume - 10.pdf
________________________________________
⚙️ Installation
1.	Clone this repository:
 	git clone https://github.com/your-username/resume-checker.git
cd resume-checker
2.	Create and activate a virtual environment (optional but recommended):
 	python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
3.	Install dependencies:
 	pip install -r requirements.txt
________________________________________
▶️ Usage
1.	Run the Streamlit app:
 	streamlit run app.py
2.	Open the local server link (default: http://localhost:8501).
3.	Authentication Password: admin
4.	Upload resumes and job descriptions for comparison.
5.	View results and analysis in the web interface.
________________________________________
🛠️ Technologies Used
•	Python
•	Streamlit (UI)
•	SQLite (Database)
•	PDF Processing Libraries (PyPDF2 / pdfminer / etc.)
•	NLP techniques (skill/keyword extraction)
________________________________________
📊 Sample Data
The project includes: - 10 sample resumes (Sample Data/Resume/) - 2 sample job descriptions (Sample Data/Job Description/)
Use these to test the application without needing to upload your own data.
________________________________________
🤝 Contribution
Contributions are welcome! Feel free to fork this repo, submit issues, or open pull requests to improve the system.
________________________________________
📜 License
This project was created as part of a Hackathon. You may modify and use it for educational or non-commercial purposes.
