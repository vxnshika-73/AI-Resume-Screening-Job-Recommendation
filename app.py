
import streamlit as st
import pdfplumber
import joblib
import re
import html

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)
# --------------------------------------------------
# PROFESSIONAL UI STYLING
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.12), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.10), transparent 28%),
        #f7f8fc;
}

/* Main content width */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit default decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Hero */
.hero {
    padding: 42px 38px;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827 0%, #312e81 55%, #4f46e5 100%);
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 20px 50px rgba(49,46,129,0.22);
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.20);
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 38px;
    line-height: 1.15;
    margin: 0 0 14px 0;
    font-weight: 800;
    color: white;
}

.hero p {
    font-size: 16px;
    line-height: 1.7;
    margin: 0;
    color: rgba(255,255,255,0.82);
    max-width: 760px;
}

/* Section heading */
.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #111827;
    margin: 28px 0 14px 0;
}

/* Upload card */
.upload-card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 30px rgba(15,23,42,0.07);
    margin-bottom: 24px;
}

.upload-icon {
    font-size: 36px;
    margin-bottom: 8px;
}

.upload-title {
    font-size: 20px;
    font-weight: 750;
    color: #111827;
}

.upload-text {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 15px;
}

/* Result cards */
.result-card {
    background: white;
    border-radius: 20px;
    padding: 25px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 30px rgba(15,23,42,0.07);
    min-height: 150px;
}

.result-label {
    color: #6b7280;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 12px;
}

.result-value {
    color: #111827;
    font-size: 27px;
    font-weight: 800;
}

.result-role {
    color: #4f46e5;
    font-size: 23px;
    font-weight: 800;
}

/* Score */
.score-number {
    font-size: 38px;
    font-weight: 800;
    color: #4f46e5;
}

.score-label {
    color: #6b7280;
    font-size: 13px;
    margin-top: 3px;
}

/* Skill chips */
.skill-chip {
    display: inline-block;
    padding: 8px 13px;
    margin: 5px 5px 5px 0;
    border-radius: 30px;
    background: #eef2ff;
    color: #3730a3;
    border: 1px solid #c7d2fe;
    font-size: 13px;
    font-weight: 600;
}

.match-chip {
    background: #ecfdf5;
    color: #047857;
    border-color: #a7f3d0;
}

.missing-chip {
    background: #fff7ed;
    color: #c2410c;
    border-color: #fed7aa;
}

/* Analysis cards */
.analysis-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 25px rgba(15,23,42,0.06);
    height: 100%;
}

.analysis-card h3 {
    margin-top: 0;
    color: #111827;
    font-size: 18px;
}

.analysis-card p {
    color: #6b7280;
    font-size: 14px;
}

/* Recommendation */
.recommendation {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    border: 1px solid #c7d2fe;
    border-radius: 18px;
    padding: 20px 22px;
    margin-top: 12px;
}

.recommendation-title {
    color: #3730a3;
    font-weight: 750;
    margin-bottom: 8px;
}

/* Footer */
.project-footer {
    margin-top: 45px;
    padding: 25px;
    text-align: center;
    border-top: 1px solid #e5e7eb;
    color: #6b7280;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)
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

# --------------------------------------------------
# PROFESSIONAL APPLICATION INTERFACE
# --------------------------------------------------

# HERO SECTION
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 AI-POWERED RESUME ANALYSIS</div>
    <h1>AI Resume Intelligence</h1>
    <p>
        Analyze your resume, discover your most suitable career role,
        evaluate your skills, identify skill gaps, and get personalized
        skill recommendations using Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# RESUME UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📄 Analyze Your Resume</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown(
        """<div style="text-align:center; padding:20px 10px 10px 10px;">
        <div style="font-size:52px;">📄</div>
        
        <h2 style="margin:10px 0 8px 0; color:#111827;">
        Upload Your Resume
        </h2>
        
        <p style="color:#6b7280; font-size:15px;">
        Upload your PDF resume and let AI analyze your skills,
        career role, and skill gaps.
        </p>
        
        <p style="color:#9ca3af; font-size:13px;">
        Supported format: PDF • Maximum size: 200 MB
        </p>
        </div>""",
                unsafe_allow_html=True
            )

    uploaded_resume = st.file_uploader(
        "Choose your resume",
        type=["pdf"],
        label_visibility="collapsed"
    )




# --------------------------------------------------
# PROCESS RESUME
# --------------------------------------------------

if uploaded_resume is not None:

    st.success(f"✓ Resume uploaded: {uploaded_resume.name}")

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

            st.success("✓ Resume text extracted successfully.")


            # --------------------------------------------------
            # RESUME PREVIEW
            # --------------------------------------------------

            with st.expander("🔍 Preview Extracted Resume Text"):

                st.text(
                    resume_text[:2000]
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
            # ANALYSIS HEADER
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">📊 AI Resume Analysis</div>',
                unsafe_allow_html=True
            )


            # --------------------------------------------------
            # TOP RESULT CARDS
            # --------------------------------------------------

            col1, col2, col3 = st.columns(3)


            with col1:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Recommended Career Role
                        </div>
                        <div class="result-role">
                            💼 {html.escape(predicted_role.replace("_", " "))}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Resume Match Score
                        </div>
                        <div class="score-number">
                            {resume_match_score:.1f}%
                        </div>
                        <div class="score-label">
                            Based on required role skills
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col3:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Skills Detected
                        </div>
                        <div class="score-number">
                            {len(detected_skills)}
                        </div>
                        <div class="score-label">
                            Skills identified in resume
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # --------------------------------------------------
            # MATCH PROGRESS
            # --------------------------------------------------

            st.markdown(
                "### 🎯 Skill Compatibility"
            )

            st.progress(
                min(int(resume_match_score), 100)
            )

            st.caption(
                f"Your resume matches approximately "
                f"{resume_match_score:.1f}% of the defined skills "
                f"for the predicted role."
            )


            # --------------------------------------------------
            # DETECTED SKILLS
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">🛠️ Detected Skills</div>',
                unsafe_allow_html=True
            )

            if detected_skills:

                skills_html = ""

                for skill in detected_skills:

                    skills_html += (
                        f'<span class="skill-chip">'
                        f'{html.escape(skill.title())}'
                        f'</span>'
                    )

                st.markdown(
                    f"""
                    <div class="analysis-card">
                        {skills_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.info(
                    "No predefined skills were detected "
                    "from the uploaded resume."
                )


            # --------------------------------------------------
            # MATCHING + MISSING SKILLS
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">🎯 Skill Gap Analysis</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)


            # MATCHING SKILLS

            with col1:

                matching_html = ""

                for skill in matching_skills:

                    matching_html += (
                        f'<span class="skill-chip match-chip">'
                        f'✓ {html.escape(skill.title())}'
                        f'</span>'
                    )


                if matching_skills:

                    st.markdown(
                        f"""
                        <div class="analysis-card">
                            <h3>✅ Matching Skills</h3>
                            <p>
                                Skills already present in your resume
                                for the predicted role.
                            </p>
                            {matching_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="analysis-card">
                            <h3>✅ Matching Skills</h3>
                            <p>No matching required skills detected.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # MISSING SKILLS

            with col2:

                missing_html = ""

                for skill in missing_skills:

                    missing_html += (
                        f'<span class="skill-chip missing-chip">'
                        f'+ {html.escape(skill.title())}'
                        f'</span>'
                    )


                if missing_skills:

                    st.markdown(
                        f"""
                        <div class="analysis-card">
                            <h3>⚠️ Skills to Improve</h3>
                            <p>
                                Skills that could strengthen your
                                profile for the predicted role.
                            </p>
                            {missing_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="analysis-card">
                            <h3>⚠️ Skills to Improve</h3>
                            <p>
                                All defined required skills were detected.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # --------------------------------------------------
            # SKILL RECOMMENDATIONS
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">💡 Personalized Recommendations</div>',
                unsafe_allow_html=True
            )


            if missing_skills:

                recommendation_items = ""

                for skill in missing_skills:

                    recommendation_items += (
                        f"""
                        <div class="recommendation">
                            <div class="recommendation-title">
                                🚀 Learn {html.escape(skill.title())}
                            </div>
                            <div>
                                Adding this skill can strengthen your
                                profile for the predicted role.
                            </div>
                        </div>
                        """
                    )


                st.markdown(
                    recommendation_items,
                    unsafe_allow_html=True
                )

            else:

                st.success(
                    "🎉 Your resume already contains all "
                    "defined required skills for this role."
                )


            # --------------------------------------------------
            # HOW THE SYSTEM WORKS
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">⚙️ How the AI System Works</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)


            with col1:

                st.markdown(
                    """
                    <div class="analysis-card">
                        <h3>1️⃣ Resume Processing</h3>
                        <p>
                            The uploaded PDF is processed and its
                            text is extracted automatically.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                st.markdown(
                    """
                    <div class="analysis-card">
                        <h3>2️⃣ ML Prediction</h3>
                        <p>
                            TF-IDF features and a trained machine
                            learning model predict a suitable job role.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col3:

                st.markdown(
                    """
                    <div class="analysis-card">
                        <h3>3️⃣ Skill Analysis</h3>
                        <p>
                            Resume skills are compared with role
                            requirements to identify skill gaps.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # --------------------------------------------------
            # PROJECT FOOTER
            # --------------------------------------------------

            st.markdown(
                """
                <div class="project-footer">
                    <strong>AI Resume Intelligence</strong><br>
                    Machine Learning • NLP • Resume Screening •
                    Job Recommendation<br><br>
                    Developed as an AI/ML major project.
                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.warning(
                "No readable text was found in the uploaded resume."
            )


    except Exception as e:

        st.error(
            "Unable to process the uploaded resume."
        )

        st.write(
            "Error:",
            e
        )
