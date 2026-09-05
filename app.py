import streamlit as st
import pdfplumber
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# NLTK SETUP
# -------------------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)


# -------------------------------------------------
# CUSTOM CSS
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 85% 5%,rgba(91,33,182,.22),transparent 25%),radial-gradient(circle at 10% 90%,rgba(37,99,235,.16),transparent 25%),#08091b;color:#f8fafc}
.main .block-container{max-width:1450px;padding-top:1.4rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c0b2a,#100b2e 55%,#09091c);border-right:1px solid rgba(139,92,246,.22)}
[data-testid="stSidebar"] *{color:#e8eaff!important}
.sidebar-brand{text-align:center;padding:8px 4px 22px}.sidebar-logo{width:58px;height:58px;margin:auto;border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:29px;background:linear-gradient(135deg,#7c3aed,#2563eb);box-shadow:0 0 30px rgba(124,58,237,.45)}
.sidebar-title{font-size:19px;font-weight:800}.sidebar-subtitle{color:#9297b8!important;font-size:11px}.sidebar-section{color:#777da5!important;font-size:10px;font-weight:800;letter-spacing:1.4px;margin:18px 0 7px 5px}.target-box{background:linear-gradient(135deg,rgba(124,58,237,.2),rgba(37,99,235,.1));border:1px solid rgba(139,92,246,.3);border-radius:16px;padding:14px}.target-label{color:#a5a9c8!important;font-size:10px;text-transform:uppercase}.target-role{font-size:18px;font-weight:800;margin-top:4px}
.hero{display:flex;align-items:center;margin-bottom:24px}.hero-left{display:flex;align-items:center;gap:14px}.hero-icon{width:56px;height:56px;border-radius:17px;display:flex;align-items:center;justify-content:center;font-size:28px;background:linear-gradient(135deg,#7c3aed,#2563eb);box-shadow:0 0 32px rgba(99,102,241,.35)}.hero-title{font-size:31px;line-height:1.05;font-weight:850}.hero-subtitle{color:#8e94b6;font-size:13px;margin-top:5px}
.panel{background:rgba(18,19,45,.82);border:1px solid rgba(139,92,246,.18);border-radius:20px;padding:20px;box-shadow:0 15px 45px rgba(0,0,0,.2)}.panel-title{font-size:16px;font-weight:800;margin-bottom:5px}.panel-help{color:#8f95b6;font-size:11px;margin-bottom:12px}
.metric-card{position:relative;overflow:hidden;min-height:130px;border-radius:19px;padding:18px;border:1px solid rgba(255,255,255,.08);box-shadow:0 14px 35px rgba(0,0,0,.22)}.metric-green{background:linear-gradient(145deg,rgba(16,185,129,.34),rgba(12,65,58,.78))}.metric-blue{background:linear-gradient(145deg,rgba(37,99,235,.38),rgba(20,35,91,.78))}.metric-purple{background:linear-gradient(145deg,rgba(124,58,237,.38),rgba(42,24,90,.78))}.metric-orange{background:linear-gradient(145deg,rgba(245,158,11,.34),rgba(86,51,16,.78))}.metric-label{color:#b7bbd1;font-size:10px;font-weight:800;letter-spacing:.8px;text-transform:uppercase}.metric-value{font-size:33px;font-weight:850;margin-top:8px;line-height:1}.metric-status{margin-top:9px;font-size:10px;color:#c7cae0}
.section-heading{font-size:19px;font-weight:850;margin:28px 0 13px}.section-heading span{color:#a78bfa}
.skill-card{background:rgba(16,17,39,.88);border:1px solid rgba(139,92,246,.16);border-radius:18px;padding:16px;box-shadow:0 12px 30px rgba(0,0,0,.16)}.skill-grid{display:flex;flex-wrap:wrap;gap:10px}.skill-pill{display:inline-flex;padding:8px 12px;border-radius:999px;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.28);color:#86efac;font-size:12px;font-weight:700}.missing-pill{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.28);color:#fbbf24}
.circle-wrap{display:flex;flex-wrap:wrap;gap:16px}.circle-item{text-align:center;width:88px}.circle{width:72px;height:72px;margin:auto;border-radius:50%;display:flex;align-items:center;justify-content:center;background:conic-gradient(#8b5cf6 var(--value),#27284a 0);position:relative}.circle:before{content:"";position:absolute;width:56px;height:56px;border-radius:50%;background:#11122d}.circle span{position:relative;z-index:2;font-size:13px;font-weight:850}.circle-name{color:#cdd0e4;font-size:10px;margin-top:7px;line-height:1.2}
.nlp-card{background:linear-gradient(145deg,rgba(20,20,52,.94),rgba(12,13,31,.96));border:1px solid rgba(34,211,238,.16);border-radius:19px;padding:16px;min-height:125px}.nlp-number{font-size:27px;font-weight:850;margin-top:4px}.nlp-label{color:#8f95b6;font-size:10px;text-transform:uppercase;letter-spacing:.8px}.nlp-ok{color:#34d399;font-weight:800;font-size:12px;margin-top:10px}.wave{height:30px;margin-top:12px;border-bottom:1px solid rgba(139,92,246,.25);position:relative;overflow:hidden}.wave:before{content:"";position:absolute;width:160%;height:28px;left:-30%;top:4px;border-radius:50%;border-top:2px solid #8b5cf6;box-shadow:0 -2px 15px rgba(139,92,246,.45)}
.recommendation{background:rgba(124,58,237,.09);border:1px solid rgba(124,58,237,.2);border-radius:12px;padding:10px 12px;margin-bottom:8px;font-size:12px;color:#d9dbea}.recommendation b{color:#a78bfa}
[data-testid="stFileUploader"]{background:rgba(10,11,30,.65);border:1px dashed rgba(139,92,246,.45);border-radius:14px;padding:6px}[data-testid="stFileUploaderDropzone"]{background:rgba(17,18,42,.75)}
textarea{background:rgba(10,11,30,.7)!important;color:#f8fafc!important;border:1px solid rgba(139,92,246,.3)!important;border-radius:13px!important}
.stButton>button{width:100%;height:48px;border:0;border-radius:13px;color:white!important;font-weight:850;font-size:14px;background:linear-gradient(90deg,#7c3aed,#2563eb);box-shadow:0 8px 25px rgba(79,70,229,.28)}.stButton>button:hover{transform:translateY(-1px);box-shadow:0 12px 32px rgba(79,70,229,.4)}
[data-baseweb="select"]>div{background:rgba(10,11,30,.72);border:1px solid rgba(139,92,246,.28);border-radius:11px}[data-testid="stExpander"]{background:rgba(16,17,39,.8);border:1px solid rgba(139,92,246,.15);border-radius:14px}hr{border-color:rgba(139,92,246,.12)}
</style>
""", unsafe_allow_html=True)

# TITLE
# -------------------------------------------------

st.markdown("""
<div class="hero"><div class="hero-left"><div class="hero-icon">🤖</div><div><div class="hero-title">AI Resume Analyzer</div><div class="hero-subtitle">Analyze • Match • Improve your resume with NLP</div></div></div></div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# SKILLS DATABASE
# -------------------------------------------------

skills_database = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "excel",
    "power bi",
    "tableau",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "machine learning",
    "deep learning",
    "nlp",
    "artificial intelligence",
    "data analysis",
    "data analytics",
    "statistics",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "streamlit",
    "git",
    "github",
    "aws",
    "azure",
    "html",
    "css",
    "javascript"
]


role_descriptions = {
    "Data Analyst": "We are looking for a Data Analyst with strong skills in Python, SQL, Excel, Power BI, Pandas, NumPy, statistics and data visualization. The candidate should be able to clean data, perform exploratory data analysis, create dashboards, identify trends and communicate business insights.",
    "Software Developer": "We are looking for a Software Developer with strong programming skills in Python, Java, C++, SQL, Git and GitHub. The candidate should be able to develop applications, debug code, work with databases and follow software development practices.",
    "Data Scientist": "We are looking for a Data Scientist with Python, SQL, Pandas, NumPy, statistics, machine learning, scikit-learn, data analysis and data visualization skills. Experience with predictive modeling is preferred.",
    "AI/ML Engineer": "We are looking for an AI/ML Engineer with Python, machine learning, deep learning, NLP, TensorFlow, PyTorch, scikit-learn, SQL and data preprocessing skills. Experience developing intelligent systems is preferred.",
    "Web Developer": "We are looking for a Web Developer with HTML, CSS, JavaScript, Python, SQL and Git skills. The candidate should be able to build responsive web applications and work with databases and APIs."
}


# -------------------------------------------------
# FUNCTION: EXTRACT PDF TEXT
# -------------------------------------------------

def extract_text_from_pdf(uploaded_file):

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# -------------------------------------------------
# FUNCTION: NLP PREPROCESSING
# -------------------------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stop word removal
    stop_words = set(stopwords.words("english"))

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# -------------------------------------------------
# FUNCTION: SKILL EXTRACTION
# -------------------------------------------------

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills_database:

        if skill.lower() in text:

            found_skills.append(skill)

    return sorted(set(found_skills))


# -------------------------------------------------
# FUNCTION: JOB MATCHING
# -------------------------------------------------

def calculate_similarity(resume_text, job_description):

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)


# -------------------------------------------------
def score_label(score):
    if score >= 80: return "Excellent"
    if score >= 65: return "Strong"
    if score >= 50: return "Good"
    return "Needs Improvement"


# SIDEBAR
# -------------------------------------------------

with st.sidebar:
    st.markdown("""<div class="sidebar-brand"><div class="sidebar-logo">🤖</div><div class="sidebar-title">AI Resume Analyzer</div><div class="sidebar-subtitle">Smart Career Intelligence</div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">NAVIGATION</div>', unsafe_allow_html=True)
    st.markdown("🏠  **Dashboard**")
    st.markdown("📄  Resume Analysis")
    st.markdown("🎯  Skill Match")
    st.markdown("🧠  NLP Insights")
    st.markdown("💡  Recommendations")
    st.markdown('<div class="sidebar-section">TARGET ROLE</div>', unsafe_allow_html=True)
    job_role = st.selectbox("Select Target Job Role", ["Data Analyst","Software Developer","Data Scientist","AI/ML Engineer","Web Developer"])
    st.markdown(f'<div class="target-box"><div class="target-label">Selected Role</div><div class="target-role">{job_role}</div></div>', unsafe_allow_html=True)


# -------------------------------------------------
# UPLOAD + JOB DESCRIPTION
# -------------------------------------------------
st.markdown('<div class="section-heading"><span>📥</span> Resume Input</div>', unsafe_allow_html=True)
input1, input2 = st.columns([.9,1.35], gap="large")
with input1:
    st.markdown('<div class="panel-title">📄 Upload Resume</div><div class="panel-help">Upload a text-based PDF resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
with input2:
    st.markdown('<div class="panel-title">💼 Job Description</div><div class="panel-help">Paste a JD or use the sample for the selected role</div>', unsafe_allow_html=True)
    use_sample = st.checkbox("Use sample job description", value=True)
    job_description = st.text_area("Job description", value=role_descriptions[job_role] if use_sample else "", height=145, label_visibility="collapsed", placeholder="Paste the job description here...")

st.markdown("")
analyze = st.button("🔍  ANALYZE RESUME", use_container_width=True)


# ANALYSIS
# -------------------------------------------------

if analyze:

    if uploaded_file is None:

        st.warning("⚠️ Please upload your resume.")

    elif job_description.strip() == "":

        st.warning("⚠️ Please enter a job description.")

    else:

        with st.spinner("Analyzing your resume..."):

            # Extract PDF text
            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text.strip():

                st.error(
                    "❌ Could not extract text from this PDF. "
                    "Please upload a text-based PDF."
                )

            else:

                # NLP preprocessing
                processed_resume = preprocess_text(
                    resume_text
                )

                processed_job = preprocess_text(
                    job_description
                )

                # Extract skills
                resume_skills = extract_skills(
                    resume_text
                )

                job_skills = extract_skills(
                    job_description
                )

                # Missing skills
                missing_skills = [
                    skill
                    for skill in job_skills
                    if skill not in resume_skills
                ]

                # Similarity
                match_score = calculate_similarity(
                    processed_resume,
                    processed_job
                )

                # ATS score
                keyword_score = 0

                if len(job_skills) > 0:

                    matched = len(
                        set(resume_skills) &
                        set(job_skills)
                    )

                    keyword_score = (
                        matched /
                        len(job_skills)
                    ) * 100

                ats_score = round(
                    (match_score + keyword_score) / 2,
                    2
                )

                # -------------------------------------------------
                # RESULTS
                # -------------------------------------------------
                tokens = word_tokenize(resume_text.lower())
                processed_tokens = word_tokenize(processed_resume)
                st.success("✅ Resume analysis completed!")

                st.markdown('<div class="section-heading"><span>📊</span> Resume Overview</div>', unsafe_allow_html=True)
                c1,c2,c3,c4=st.columns(4,gap="medium")
                cards=[
                    (c1,"metric-green","ATS SCORE",f"{ats_score}%",score_label(ats_score)),
                    (c2,"metric-blue","JOB MATCH",f"{match_score}%",score_label(match_score)+" Match"),
                    (c3,"metric-purple","SKILLS FOUND",str(len(resume_skills)),"Detected from resume"),
                    (c4,"metric-orange","MISSING SKILLS",str(len(missing_skills)),"Good coverage" if not missing_skills else "Improve these")
                ]
                for col,theme,label,value,status in cards:
                    with col:
                        st.markdown(f'<div class="metric-card {theme}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-status">● {status}</div></div>',unsafe_allow_html=True)

                left,right=st.columns([1.5,1],gap="large")
                with left:
                    st.markdown('<div class="section-heading"><span>🎯</span> Skill Match Analysis</div>',unsafe_allow_html=True)
                    if job_skills:
                        circles=""
                        for skill in job_skills[:10]:
                            value=100 if skill in resume_skills else 0
                            circles+=f'<div class="circle-item"><div class="circle" style="--value:{value}%"><span>{value}%</span></div><div class="circle-name">{skill.title()}</div></div>'
                        st.markdown(f'<div class="skill-card"><div class="circle-wrap">{circles}</div></div>',unsafe_allow_html=True)
                    else:
                        st.info("No predefined skills detected in the job description.")
                with right:
                    st.markdown('<div class="section-heading"><span>🧠</span> NLP Analysis</div>',unsafe_allow_html=True)
                    n1,n2=st.columns(2)
                    with n1:
                        st.markdown(f'<div class="nlp-card"><div class="nlp-label">Tokens</div><div class="nlp-number">{len(tokens)}</div><div class="nlp-ok">✓ Tokenized</div></div>',unsafe_allow_html=True)
                    with n2:
                        st.markdown(f'<div class="nlp-card"><div class="nlp-label">Processed Words</div><div class="nlp-number">{len(processed_tokens)}</div><div class="nlp-ok">✓ Cleaned</div></div>',unsafe_allow_html=True)
                    st.markdown(f'<div class="nlp-card" style="margin-top:12px"><div class="nlp-label">TF-IDF + Cosine Similarity</div><div class="nlp-number">{match_score}%</div><div class="nlp-ok">✓ TF-IDF &nbsp;&nbsp; ✓ Cosine Similarity</div><div class="wave"></div></div>',unsafe_allow_html=True)

                a,b=st.columns(2,gap="large")
                with a:
                    st.markdown('<div class="section-heading"><span>✅</span> Skills Found</div>',unsafe_allow_html=True)
                    if resume_skills:
                        html='<div class="skill-card"><div class="skill-grid">'+''.join(f'<span class="skill-pill">✓ {x.title()}</span>' for x in resume_skills)+'</div></div>'
                        st.markdown(html,unsafe_allow_html=True)
                    else: st.info("No skills detected.")
                with b:
                    st.markdown('<div class="section-heading"><span>⚠️</span> Missing Skills</div>',unsafe_allow_html=True)
                    if missing_skills:
                        html='<div class="skill-card"><div class="skill-grid">'+''.join(f'<span class="skill-pill missing-pill">+ {x.title()}</span>' for x in missing_skills)+'</div></div>'
                        st.markdown(html,unsafe_allow_html=True)
                    else: st.success("🎉 Your resume contains all detected job skills!")

                st.markdown('<div class="section-heading"><span>💡</span> Personalized Recommendations</div>',unsafe_allow_html=True)
                recs=[]
                if missing_skills: recs.append("Add relevant skills: "+", ".join(x.title() for x in missing_skills[:5]))
                if match_score<50: recs.append("Your resume has a low text similarity. Add relevant job keywords naturally.")
                elif match_score<75: recs.append("Your resume has a moderate match. Strengthen project descriptions with job-related keywords.")
                else: recs.append("Your resume has a strong match. Keep the important keywords and add measurable achievements.")
                recs.append("Use measurable results in projects, such as percentages, records analyzed, time saved, or accuracy.")
                for rec in recs: st.markdown(f'<div class="recommendation">✦ <b>Tip:</b> {rec}</div>',unsafe_allow_html=True)

                with st.expander("📄 View Extracted Resume Text"):
                    st.text_area("Extracted text",resume_text,height=260,label_visibility="collapsed")

                st.caption("AI Resume Analyzer • NLP preprocessing • Skill extraction • TF-IDF • Cosine Similarity")
