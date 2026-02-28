# =========================================================
# Student Learning Analysis - Simple Version
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Student Analysis", page_icon="📊", layout="centered")

st.title("📊 Student Learning Pattern Analysis")
st.divider()

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ---------------------------------------------------------
# ANALYSIS FUNCTION
# ---------------------------------------------------------
def analyze(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()

    for c in ["g1", "g2", "g3"]:
        if c not in df.columns:
            st.error(f"CSV must have column: {c.upper()}")
            return None

    df["g1"] = pd.to_numeric(df["g1"], errors="coerce")
    df["g2"] = pd.to_numeric(df["g2"], errors="coerce")
    df["g3"] = pd.to_numeric(df["g3"], errors="coerce")
    df = df.dropna(subset=["g1","g2","g3"]).reset_index(drop=True)

    if len(df) < 5:
        st.error("Need at least 5 valid students.")
        return None

    # Features
    df["avg_grade"]      = df[["g1","g2","g3"]].mean(axis=1)
    df["grade_trend"]    = df["g3"] - df["g1"]
    df["grade_variance"] = df[["g1","g2","g3"]].var(axis=1)
    df["grade_stability"]= 1 / (1 + df["grade_variance"].fillna(0.1))

    def gc(name, default):
        return df[name].astype(float) if name in df.columns else pd.Series([default]*len(df), index=df.index)
    def yn(name):
        return df[name].map({"yes":1,"no":0}).fillna(0) if name in df.columns else pd.Series([0]*len(df), index=df.index)

    absences   = gc("absences", 0)
    studytime  = gc("studytime", 2)
    traveltime = gc("traveltime", 1)

    df["study_discipline"] = ((studytime/4)*0.35 + (1-traveltime/4)*0.25 + (1-absences/(absences.max()+1))*0.40).fillna(0.5)
    df["engagement"]       = (yn("activities")*0.25 + yn("higher")*0.20 + yn("schoolsup")*0.20 + yn("internet")*0.20 + yn("paid")*0.15).fillna(0.3)

    medu = gc("medu", 2); fedu = gc("fedu", 2); famrel = gc("famrel", 3)
    df["family_support"]   = ((medu+fedu)/8*0.35 + yn("famsup")*0.30 + (famrel/5)*0.35).fillna(0.4)

    goout = gc("goout",2); dalc = gc("dalc",1); walc = gc("walc",1)
    df["lifestyle_risk"]   = ((goout/5)*0.30 + (dalc/5)*0.35 + (walc/5)*0.35).fillna(0.2)

    health = gc("health", 3)
    df["health_score"]     = (1 - health/5).fillna(0.4)

    pstatus = df["pstatus"].str.lower().map({"t":0,"a":1}).fillna(0) if "pstatus" in df.columns else pd.Series([0.0]*len(df), index=df.index)
    df["low_ses"] = (pstatus*0.5 + (1-medu/4)*0.25 + (1-fedu/4)*0.25).fillna(0.3)

    # Clustering
    features = ["avg_grade","grade_stability","grade_trend","study_discipline",
                "engagement","family_support","lifestyle_risk","health_score","low_ses"]
    X = StandardScaler().fit_transform(df[features].fillna(0))

    best_k, best_sil = 3, -1
    from sklearn.metrics import silhouette_score
    for k in range(2, min(8, len(df))):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        s = silhouette_score(X, labels)
        if s > best_sil: best_sil, best_k = s, k

    df["cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)

    # Persona
    persona_map = {}
    for cid in range(best_k):
        cdf = df[df["cluster"]==cid]
        ag = cdf["avg_grade"].mean(); eng = cdf["engagement"].mean()
        dis = cdf["study_discipline"].mean()
        ab  = cdf["absences"].mean() if "absences" in df.columns else 5
        risk= cdf["lifestyle_risk"].mean()

        if   ag >= 15: persona = "🌟 High Achiever" if dis >= 0.6 else "💎 High Potential"
        elif ag >= 12:
            if   dis >= 0.5 and eng >= 0.5: persona = "✅ Solid Performer"
            elif ab >= 15:                   persona = "⚠️ Attendance Issues"
            else:                            persona = "📈 Above Average"
        elif ag >= 10:
            if   eng >= 0.5: persona = "🔄 Developing Learner"
            elif dis >= 0.5: persona = "💪 Struggling but Trying"
            else:            persona = "⚠️ Low Engagement"
        else:
            if   ab >= 20:   persona = "🔴 At-Risk (Absenteeism)"
            elif eng < 0.3:  persona = "🔴 At-Risk (Disengaged)"
            elif risk > 0.6: persona = "🔴 At-Risk (Lifestyle)"
            else:            persona = "🔴 Struggling"
        persona_map[cid] = persona

    df["persona_label"] = df["cluster"].map(persona_map)

    # Risk
    def detect_risks(row):
        risks, scores = [], []
        ag = row["avg_grade"]; gt = row["grade_trend"]
        ab  = row["absences"]  if "absences"  in row.index else 0
        stt = row["studytime"] if "studytime" in row.index else 2

        if   ag < 8:             risks.append("🔴 Critical – Failing grades");      scores.append(0.95)
        elif ag < 10 and gt < 0: risks.append("🔴 Severe – Declining grades");      scores.append(0.85)
        elif ag < 10:            risks.append("🟠 Moderate – Low grades (<10)");    scores.append(0.70)
        elif gt < -3:            risks.append("🟠 Moderate – Sharp grade decline"); scores.append(0.75)
        elif gt < -1 and ab > 5: risks.append("🟡 Low – Declining + absences");     scores.append(0.55)

        if   ab > 25: risks.append("🔴 Critical – Chronic absenteeism"); scores.append(0.90)
        elif ab > 15: risks.append("🟠 Moderate – High absences");       scores.append(0.65)
        elif ab > 8:  risks.append("🟡 Low – Notable absences");         scores.append(0.45)

        if   stt <= 1 and ag < 10: risks.append("🔴 Critical – No study + failing"); scores.append(0.88)
        elif stt <= 2 and ag < 12: risks.append("🟠 Moderate – Minimal study");      scores.append(0.68)

        if row["engagement"]     < 0.2: risks.append("🟠 Moderate – Very low engagement"); scores.append(0.70)
        if row["lifestyle_risk"] > 0.7: risks.append("🟠 Moderate – High lifestyle risk");  scores.append(0.65)

        if not risks:
            risks.append("✅ No major risks detected"); scores.append(0)

        return risks, float(np.mean(scores))

    results = df.apply(lambda r: pd.Series(detect_risks(r)), axis=1)
    df["risk_flags"]  = results[0]
    df["risk_score"]  = results[1]

    # Strategies
    STRAT = {
        "🌟 High Achiever":         ["Advanced enrichment & projects", "Peer tutoring/mentoring roles", "Prepare for competitive higher education"],
        "💎 High Potential":        ["Structured time management", "Firm deadlines with accountability", "One-on-one mentoring"],
        "✅ Solid Performer":        ["Continue current support", "Gradually increase challenge", "Career planning guidance"],
        "⚠️ Attendance Issues":     ["Parent-teacher meeting on attendance", "Identify barriers to attendance", "Attendance incentive plan"],
        "📈 Above Average":         ["Consistent encouragement", "Explore advanced topics", "Set stretch goals"],
        "🔄 Developing Learner":    ["Scaffolded assignments", "Regular formative feedback", "Achievable milestone goals"],
        "💪 Struggling but Trying": ["Validate effort & progress", "Diagnostic skill assessments", "Targeted tutoring"],
        "⚠️ Low Engagement":        ["Investigate disengagement causes", "Connect to student interests", "Family involvement plan"],
        "🔴 At-Risk (Absenteeism)": ["URGENT: Attendance intervention", "School counselor coordination", "Daily check-in system"],
        "🔴 At-Risk (Disengaged)":  ["Priority parent conference", "Assign academic mentor", "Daily engagement monitoring"],
        "🔴 At-Risk (Lifestyle)":   ["Confidential health discussion", "Counselor/social worker referral", "Weekly progress check-ins"],
        "🔴 Struggling":            ["Intensive tutoring program", "Daily learning check-ins", "Multi-agency support"],
    }
    df["teaching_strategy"] = df["persona_label"].map(lambda p: STRAT.get(p, ["Provide general support"]))

    return df


# ---------------------------------------------------------
# STEP 1 — UPLOAD CSV
# ---------------------------------------------------------
uploaded = st.file_uploader("📁 Step 1: Upload your student dataset (CSV)", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    with st.spinner("Analyzing all students... please wait"):
        result = analyze(raw)
    if result is not None:
        st.session_state.df = result
        st.session_state.analyzed = True
        st.success(f"✅ Done! **{len(result)} students** analyzed successfully.")

# ---------------------------------------------------------
# STEP 2 — SHOW FULL DATASET TABLE
# ---------------------------------------------------------
if st.session_state.analyzed and st.session_state.df is not None:
    df = st.session_state.df

    st.divider()
    st.subheader("📋 Step 2: Full Student Dataset")

    display_cols = ["g1","g2","g3","avg_grade","grade_trend","persona_label","risk_score"]
    if "absences" in df.columns:
        display_cols.insert(5, "absences")

    st.dataframe(
        df[display_cols].rename(columns={
            "g1":"G1", "g2":"G2", "g3":"G3",
            "avg_grade":"Avg Grade", "grade_trend":"Trend",
            "absences":"Absences", "persona_label":"Persona",
            "risk_score":"Risk Score"
        }),
        use_container_width=True,
        height=300
    )

    # ---------------------------------------------------------
    # STEP 3 — ENTER STUDENT NUMBER & GET ANALYSIS
    # ---------------------------------------------------------
    st.divider()
    st.subheader("🔍 Step 3: Enter Student Number for Detailed Analysis")

    col_input, col_btn = st.columns([2, 1])
    with col_input:
        student_no = st.number_input(
            f"Student Number (0 to {len(df)-1})",
            min_value=0,
            max_value=len(df)-1,
            step=1,
            value=0,
            label_visibility="collapsed"
        )
    with col_btn:
        analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)

    if analyse_btn:
        s = df.loc[int(student_no)]
        rs = float(s["risk_score"])
        rc = "#FF4444" if rs > 0.7 else ("#FFA500" if rs >= 0.4 else "#90EE90")
        risk_label = "🔴 HIGH RISK" if rs > 0.7 else ("🟠 MEDIUM RISK" if rs >= 0.4 else "🟢 LOW RISK")

        st.divider()
        st.subheader(f"📌 Student {int(student_no)} — Detailed Report")

        # Top card
        st.markdown(f"""
        <div style="background:#1a1a2e; border:2px solid {rc}; border-radius:12px; padding:20px; margin-bottom:20px;">
            <h2 style="color:white; margin:0 0 10px 0;">Student {int(student_no)}</h2>
            <p style="font-size:1.1rem; margin:6px 0;">
                🎭 <b>Persona:</b> <span style="color:#00BFFF; font-size:1.2rem;">{s['persona_label']}</span>
            </p>
            <p style="font-size:1.1rem; margin:6px 0;">
                ⚠️ <b>Risk Score:</b>
                <span style="color:{rc}; font-weight:bold; font-size:1.3rem;"> {rs:.2f} — {risk_label}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Grades
        st.markdown("#### 📚 Grades")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("G1",      f"{s.get('g1', 'N/A')}")
        c2.metric("G2",      f"{s.get('g2', 'N/A')}")
        c3.metric("G3",      f"{s.get('g3', 'N/A')}")
        c4.metric("Average", f"{float(s['avg_grade']):.2f}")
        c5.metric("Trend",   f"{float(s['grade_trend']):+.2f}")

        # Study & Behaviour
        st.markdown("#### 📊 Study & Behaviour")
        d1, d2, d3 = st.columns(3)
        d1.metric("Study Discipline", f"{float(s['study_discipline']):.2f}")
        d2.metric("Engagement",       f"{float(s['engagement']):.2f}")
        d3.metric("Absences",         f"{int(s['absences']) if 'absences' in s.index else 'N/A'}")

        # Family & Lifestyle
        st.markdown("#### 👨‍👩‍👧 Family & Lifestyle")
        e1, e2, e3 = st.columns(3)
        e1.metric("Family Support",  f"{float(s['family_support']):.2f}")
        e2.metric("Lifestyle Risk",  f"{float(s['lifestyle_risk']):.2f}")
        e3.metric("Health Score",    f"{float(s['health_score']):.2f}")

        # Risk Flags
        st.markdown("#### 🚩 Risk Flags")
        flags = s["risk_flags"] if isinstance(s["risk_flags"], list) else [str(s["risk_flags"])]
        for flag in flags:
            color = "#FF4444" if "🔴" in flag else ("#FFA500" if "🟠" in flag else ("#FFD700" if "🟡" in flag else "#90EE90"))
            st.markdown(f'<p style="color:{color}; font-size:1rem; margin:4px 0; padding:4px 8px; background:#ffffff11; border-radius:6px;">• {flag}</p>', unsafe_allow_html=True)

        # Teaching Strategies
        st.markdown("#### 💡 Recommended Teaching Strategies")
        strats = s["teaching_strategy"] if isinstance(s["teaching_strategy"], list) else [str(s["teaching_strategy"])]
        for strat in strats:
            st.markdown(f'<p style="color:#FFD700; font-size:1rem; margin:4px 0; padding:4px 8px; background:#ffffff11; border-radius:6px;">✅ {strat}</p>', unsafe_allow_html=True)