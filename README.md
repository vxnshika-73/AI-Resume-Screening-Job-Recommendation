# 🤖 AI Resume Screening & Job Recommendation System

An AI and NLP-based Resume Screening and Job Recommendation System that automates resume analysis, predicts suitable job roles, identifies candidate skills, performs skill-gap analysis, and provides personalized skill recommendations.

---

## 📌 Project Overview

Recruitment processes often involve reviewing a large number of resumes for different job positions. Manual resume screening can be time-consuming and may make it difficult to consistently identify relevant skills and job roles.

The **AI Resume Screening & Job Recommendation System** addresses this problem by using **Machine Learning and Natural Language Processing (NLP)** techniques to analyze resume content and provide automated insights.

The system extracts text from uploaded PDF resumes, preprocesses the text, converts it into numerical features using **TF-IDF**, and uses a trained Machine Learning model to predict the most suitable job category.

It also analyzes the skills present in the resume, compares them with role-specific skills, identifies missing skills, and provides recommendations for improvement.

---

## 🎯 Objectives

The main objectives of this project are:

- Automate the initial resume screening process.
- Extract meaningful information from resume documents.
- Predict suitable job roles using Machine Learning.
- Identify technical and professional skills from resumes.
- Perform skill-gap analysis.
- Identify missing skills for the predicted role.
- Provide skill improvement recommendations.
- Calculate a project-defined resume skill-match score.
- Develop an interactive and user-friendly web application.

---

## 💡 Key Features

### 📄 Resume Processing
- Upload resumes in PDF format.
- Extract resume text using PDFPlumber.
- Clean and preprocess extracted text.

### 🧠 Job Role Prediction
- Convert resume text into TF-IDF features.
- Predict the most suitable job category using a trained ML model.

### 🔍 Skill Extraction
The system detects relevant skills such as:

- Python
- Java
- C++
- JavaScript
- HTML
- CSS
- React
- Node.js
- SQL
- MySQL
- MongoDB
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- PyTorch
- Machine Learning
- Deep Learning
- NLP
- Git
- GitHub
- Docker
- AWS
- Azure
- Power BI
- Tableau
- Excel

### 📊 Skill-Gap Analysis
The system compares detected resume skills with predefined skills required for the predicted role.

It identifies:

- Matching skills
- Missing skills
- Skill-match percentage
- Recommended skills for improvement

### 📈 Resume Match Score

The application provides a **skill-based resume match score** to indicate how closely the detected skills align with the requirements defined for the predicted role.

> **Note:** This score is a project-defined skill matching indicator. It is not a validated hiring score and should not be used as the sole basis for recruitment decisions.

### 🌐 Interactive Web Application

The system is implemented using **Streamlit**, providing an interactive interface where users can upload a resume and view the analysis.

---

# 🏗️ System Architecture

```text
                ┌──────────────────────┐
                │     Resume PDF       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Text Extraction    │
                │     PDFPlumber       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Text Preprocessing   │
                │ Cleaning & Normalize │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ TF-IDF Vectorization │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   ML Classification  │
                │ Logistic Regression  │
                │ Decision Tree        │
                │ Random Forest        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Job Role Prediction│
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌──────────────────┐      ┌──────────────────┐
     │ Skill Extraction │      │ Role Requirements│
     └────────┬─────────┘      └────────┬─────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                ┌──────────────────────┐
                │  Skill Gap Analysis  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Recommendations &    │
                │ Match Score          │
                └──────────────────────┘
