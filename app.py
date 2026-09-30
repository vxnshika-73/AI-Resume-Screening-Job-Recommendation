
import streamlit as st
import pdfplumber
import joblib
import re

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="centered"
)

# --------------------------------------------------
# LOAD TRAINED ML COMPONENTS
# --------------------------------------------------

model = joblib.load("models/best_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# --------------------------------------------------
# RESUME CLEANING FUNCTION
# --------------------------------------------------

def clean_resume(text):

    text = str(text)
    text = text.lower()

    text = re.sub(
        r'http\S+|www\S+',
        ' ',
        text
    )

    text = re.sub(
        r'\S+@\S+',
        ' ',
        text
    )

    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# --------------------------------------------------
# SKILLS LIST
# --------------------------------------------------

skills_list = [
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "angular",
    "node.js",
    "flask",
    "django",
    "fastapi",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "keras",
    "pytorch",
    "machine learning",
    "deep learning",
    "nlp",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "power bi",
    "tableau",
    "excel"
]


# --------------------------------------------------
# SKILL EXTRACTION FUNCTION
# --------------------------------------------------

def extract_skills(text):

    text_lower = text.lower()

    detected_skills = []

    for skill in skills_list:

        if skill.lower() in text_lower:
            detected_skills.append(skill)

    return sorted(set(detected_skills))


# --------------------------------------------------
# REQUIRED SKILLS FOR EACH ROLE
# --------------------------------------------------

role_required_skills = {

    "Python_Developer": [
        "python",
        "sql",
        "pandas",
        "numpy",
        "machine learning",
        "git",
        "flask"
    ],

    "Java_Developer": [
        "java",
        "sql",
        "git",
        "javascript",
        "html",
        "css"
    ],

    "Web_Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "node.js",
        "git"
    ],

    "Front_End_Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "angular"
    ],

    "Software_Developer": [
        "python",
        "java",
        "sql",
        "git",
        "javascript"
    ],

    "Database_Administrator": [
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "python"
    ],

    "Network_Administrator": [
        "networking",
        "linux",
        "tcp/ip",
        "firewall",
        "git"
    ],

    "Security_Analyst": [
        "cybersecurity",
        "networking",
        "linux",
        "firewall",
        "python"
    ],

    "Systems_Administrator": [
        "linux",
        "python",
        "networking",
        "docker",
        "aws"
    ],

    "Project_manager": [
        "excel",
        "power bi",
        "tableau",
        "git",
        "sql"
    ]
}


# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.title(
    "AI Resume Screening & Job Recommendation System"
)

st.write(
    "Upload your resume to predict a suitable job role, "
    "analyze your skills, and identify skill gaps."
)

st.divider()


# --------------------------------------------------
# RESUME UPLOAD
# --------------------------------------------------

st.subheader("📄 Upload Resume")

uploaded_resume = st.file_uploader(
    "Choose your resume",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS RESUME
# --------------------------------------------------

if uploaded_resume is not None:

    st.success("Resume uploaded successfully!")

    st.write(
        "File Name:",
        uploaded_resume.name
    )

    resume_text = ""

    try:

        # --------------------------------------------------
        # EXTRACT PDF TEXT
        # --------------------------------------------------

        with pdfplumber.open(uploaded_resume) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if text:

                    resume_text += text + "\n"


        if resume_text.strip():

            st.success(
                "Resume text extracted successfully."
            )


            # --------------------------------------------------
            # RESUME PREVIEW
            # --------------------------------------------------

            with st.expander(
                "Preview Extracted Resume Text"
            ):

                st.text(
                    resume_text[:1500]
                )


            # --------------------------------------------------
            # JOB ROLE PREDICTION
            # --------------------------------------------------

            cleaned_text = clean_resume(
                resume_text
            )

            resume_vector = (
                tfidf_vectorizer.transform(
                    [cleaned_text]
                )
            )

            prediction = model.predict(
                resume_vector
            )

            predicted_role = (
                label_encoder.inverse_transform(
                    prediction
                )[0]
            )


            # --------------------------------------------------
            # SKILL DETECTION
            # --------------------------------------------------

            detected_skills = extract_skills(
                resume_text
            )


            # --------------------------------------------------
            # REQUIRED SKILLS
            # --------------------------------------------------

            required_skills = (
                role_required_skills.get(
                    predicted_role,
                    []
                )
            )


            # --------------------------------------------------
            # SKILL MATCH CALCULATION
            # --------------------------------------------------

            detected_lower = [
                skill.lower()
                for skill in detected_skills
            ]

            matching_skills = []

            missing_skills = []


            for skill in required_skills:

                if skill.lower() in detected_lower:

                    matching_skills.append(
                        skill
                    )

                else:

                    missing_skills.append(
                        skill
                    )


            if required_skills:

                skill_match_percentage = (
                    len(matching_skills)
                    / len(required_skills)
                ) * 100

            else:

                skill_match_percentage = 0


            # --------------------------------------------------
            # RESUME MATCH SCORE
            # --------------------------------------------------

            resume_match_score = round(
                skill_match_percentage,
                2
            )


            # --------------------------------------------------
            # RESULTS
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "📊 Resume Analysis"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Predicted Job Role",
                    predicted_role
                )


            with col2:

                st.metric(
                    "Resume Match Score",
                    f"{resume_match_score:.2f}%"
                )


            st.progress(
                int(resume_match_score)
            )


            # --------------------------------------------------
            # DETECTED SKILLS
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "🛠️ Detected Skills"
            )


            if detected_skills:

                st.write(
                    ", ".join(
                        skill.title()
                        for skill in detected_skills
                    )
                )

            else:

                st.info(
                    "No predefined skills were detected."
                )


            # --------------------------------------------------
            # MATCHING SKILLS
            # --------------------------------------------------

            st.subheader(
                "✅ Matching Skills"
            )


            if matching_skills:

                st.write(
                    ", ".join(
                        skill.title()
                        for skill in matching_skills
                    )
                )

            else:

                st.info(
                    "No matching skills detected."
                )


            # --------------------------------------------------
            # MISSING SKILLS
            # --------------------------------------------------

            st.subheader(
                "⚠️ Missing Skills"
            )


            if missing_skills:

                for skill in missing_skills:

                    st.markdown(
                        f"- **{skill.title()}**"
                    )

            else:

                st.success(
                    "All defined required skills were detected!"
                )


            # --------------------------------------------------
            # SKILL RECOMMENDATIONS
            # --------------------------------------------------

            st.subheader(
                "💡 Skill Recommendations"
            )


            if missing_skills:

                st.write(
                    "Consider learning these skills "
                    "to strengthen your profile:"
                )

                for skill in missing_skills:

                    st.markdown(
                        f"➡️ **{skill.title()}**"
                    )

            else:

                st.success(
                    "No additional skills are currently "
                    "recommended based on the defined role requirements."
                )


        else:

            st.warning(
                "No readable text was found "
                "in the uploaded resume."
            )


    except Exception as e:

        st.error(
            "Unable to process the uploaded resume."
        )

        st.write(
            "Error:",
            e
        )
