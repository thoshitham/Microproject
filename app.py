
import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

st.title("AI Resume Matcher & Career Recommendation System")
st.write("Upload your resume and paste a job description.")

SKILLS = [
    "python", "java", "c++", "c", "sql", "mysql", "html",
    "css", "javascript", "machine learning", "deep learning",
    "artificial intelligence", "data analysis", "pandas",
    "numpy", "tensorflow", "pytorch", "nlp", "computer vision",
    "power bi", "excel", "git", "communication", "teamwork",
    "problem solving", "aws", "flask", "django", "statistics",
    "generative ai", "llm", "prompt engineering"
]


def extract_resume_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = []

    for page in reader.pages:
        text.append(page.extract_text() or "")

    return "\n".join(text).strip()


def find_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def calculate_match(resume, job_description):
    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(
        [resume, job_description]
    )

    score = cosine_similarity(
        vectors[0:1], vectors[1:2]
    )[0][0]

    return round(float(score) * 100, 2)


resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

job_title = st.text_input(
    "Enter job title",
    placeholder="Example: AI/ML Intern"
)

job_description = st.text_area(
    "Paste the Job Description",
    height=200
)

if st.button("Analyze Resume", type="primary"):

    if resume_file is None:
        st.error("Please upload your resume.")

    elif not job_description.strip():
        st.error("Please enter the job description.")

    else:
        try:
            resume_text = extract_resume_text(resume_file)

            if len(resume_text) < 30:
                st.error(
                    "Could not read enough text. "
                    "Please upload a text-based PDF."
                )
                st.stop()

            score = calculate_match(
                resume_text,
                job_description
            )

            resume_skills = find_skills(resume_text)
            job_skills = find_skills(job_description)

            matched_skills = sorted(
                set(resume_skills) & set(job_skills)
            )

            missing_skills = sorted(
                set(job_skills) - set(resume_skills)
            )

            st.divider()
            st.header("Resume Analysis Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Match Score", f"{score}%")
            col2.metric("Matching Skills", len(matched_skills))
            col3.metric("Missing Skills", len(missing_skills))

            st.progress(int(score) / 100)

            st.info(
                "The match score measures text similarity. "
                "It is not the probability of getting hired."
            )

            left, right = st.columns(2)

            with left:
                st.subheader("Matching Skills")

                if matched_skills:
                    for skill in matched_skills:
                        st.write("✅ " + skill)
                else:
                    st.write("No matching skills detected.")

            with right:
                st.subheader("Skills to Learn")

                if missing_skills:
                    for skill in missing_skills:
                        st.write("📌 " + skill)
                else:
                    st.write("No missing skills detected.")

            st.divider()
            st.header("Career Recommendations")

            st.write(
                "Based on your selected job, "
                f"you are exploring the role of {job_title or 'your target job'}."
            )

            if missing_skills:
                st.write(
                    "Focus on learning: "
                    + ", ".join(missing_skills)
                )

                st.write("Suggested learning plan:")

                st.write("Week 1: Learn the basics.")
                st.write("Week 2: Practise with exercises.")
                st.write("Week 3: Build a small project.")
                st.write("Week 4: Add the project to your resume.")

            else:
                st.write(
                    "Build a project that demonstrates the "
                    "important skills in the job description."
                )

            st.subheader("Resume Improvement Tips")

            st.write("1. Highlight relevant technical skills.")
            st.write("2. Explain your projects clearly.")
            st.write("3. Add measurable achievements when available.")
            st.write("4. Include job keywords that match your real experience.")

            report = f"""
RESUME ANALYSIS REPORT

Job Title: {job_title or 'Not provided'}
Match Score: {score}%

Matching Skills:
{', '.join(matched_skills) or 'None detected'}

Missing Skills:
{', '.join(missing_skills) or 'None detected'}
"""

            st.download_button(
                "Download Analysis Report",
                data=report,
                file_name="resume_analysis.txt",
                mime="text/plain"
            )

        except Exception as error:
            st.error(f"Error: {error}")

st.divider()
st.caption("Generative AI Resume Matcher - Student Project Prototype")
