# 📊 Student Learning Pattern Analysis

A simple Streamlit web application that analyzes student academic data and generates meaningful insights such as learning personas, risk levels, and recommended teaching strategies.

This project is designed to help teachers or institutions quickly understand student performance patterns using data.

---

## 🚀 What This App Does

After uploading a student dataset (CSV file), the app:

- Calculates average grades and grade trends
- Measures study discipline and engagement levels
- Detects possible academic or behavioral risks
- Groups students using clustering (K-Means)
- Assigns each student a learning persona
- Suggests teaching strategies based on their profile

---

## 📁 Required Dataset Format

Your CSV file must contain the following columns:

- `G1` – First period grade  
- `G2` – Second period grade  
- `G3` – Final grade  

Optional but recommended columns:

- `absences`
- `studytime`
- `traveltime`
- `activities`
- `higher`
- `schoolsup`
- `internet`
- `paid`
- `medu`
- `fedu`
- `famrel`
- `goout`
- `dalc`
- `walc`
- `health`
- `pstatus`

At least 5 valid student records are required.

---

## 🧠 How It Works

### 1️⃣ Feature Engineering
The app automatically calculates:

- Average grade
- Grade trend (improving or declining)
- Grade stability
- Study discipline score
- Engagement score
- Family support score
- Lifestyle risk score
- Health score
- Socioeconomic risk indicator

### 2️⃣ Clustering
Students are grouped using:
- StandardScaler
- K-Means clustering
- Silhouette Score (to select optimal number of clusters)

### 3️⃣ Persona Generation
Each cluster is assigned a persona such as:

- 🌟 High Achiever
- 💎 High Potential
- ✅ Solid Performer
- 📈 Above Average
- 🔄 Developing Learner
- ⚠️ Attendance Issues
- 🔴 At-Risk

### 4️⃣ Risk Detection
Each student receives:
- Risk flags
- A risk score (0 to 1 scale)
- Risk level (Low / Medium / High)

### 5️⃣ Teaching Strategies
Based on the persona and risk level, the app recommends actionable teaching strategies.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

## 🎯 Purpose of This Project

This project was built to explore how machine learning and data analysis can be applied in the education domain. It demonstrates:

- Data preprocessing
- Feature engineering
- Clustering
- Risk scoring logic
- Dashboard development with Streamlit

---


