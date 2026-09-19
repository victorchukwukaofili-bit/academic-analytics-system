import sqlite3
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(
    page_title="Academic Early Warning & Management System",
    page_icon="🎓",
    layout="wide",
)


# --- DATABASE & MODEL HELPERS ---
@st.cache_resource
def load_model():
  return joblib.load("model.pkl")


def get_db_connection():
  return sqlite3.connect("database.db")


def load_students():
  conn = get_db_connection()
  df = pd.read_sql_query("SELECT * FROM students", conn)
  conn.close()
  return df


def load_student_courses(matric_number):
  conn = get_db_connection()
  query = """
        SELECT course_code AS 'Course Code', 
               course_title AS 'Course Title', 
               ca_score_30 AS 'CA Score (/30)' 
        FROM student_courses 
        WHERE matric_number = ?
    """
  df_courses = pd.read_sql_query(query, conn, params=(matric_number,))
  conn.close()

  df_courses["Status"] = df_courses["CA Score (/30)"].apply(
      lambda score: "🔴 Failing CA (< 12)" if score < 12 else "🟢 On Track"
  )
  return df_courses


def authenticate_user(username, password, role):
  conn = get_db_connection()
  cursor = conn.cursor()
  cursor.execute(
      """
        SELECT username, role, full_name FROM users 
        WHERE LOWER(username) = LOWER(?) AND password = ? AND role = ?
    """,
      (username.strip(), password.strip(), role),
  )
  user = cursor.fetchone()
  conn.close()
  return user


def register_user(username, password, role, full_name):
  conn = get_db_connection()
  cursor = conn.cursor()
  try:
    cursor.execute(
        """
            INSERT INTO users (username, password, role, full_name)
            VALUES (?, ?, ?, ?)
        """,
        (username.strip().upper(), password.strip(), role, full_name.strip()),
    )
    conn.commit()
    conn.close()
    return True, "Account created successfully! You can now log in."
  except sqlite3.IntegrityError:
    conn.close()
    return False, f"Username / ID '{username}' already exists."


def update_student_ca_and_repredict(matric_number, course_code, new_ca_score):
  conn = get_db_connection()
  cursor = conn.cursor()

  cursor.execute(
      """
        UPDATE student_courses 
        SET ca_score_30 = ? 
        WHERE matric_number = ? AND course_code = ?
    """,
      (new_ca_score, matric_number, course_code),
  )

  scores_df = pd.read_sql_query(
      "SELECT ca_score_30 FROM student_courses WHERE matric_number = ?",
      conn,
      params=(matric_number,),
  )
  ca_scores = scores_df["ca_score_30"].values

  new_ca_mean = float(np.mean(ca_scores))
  new_ca_var = float(np.var(ca_scores)) if len(ca_scores) > 1 else 0.0

  student_df = pd.read_sql_query(
      "SELECT * FROM students WHERE matric_number = ?",
      conn,
      params=(matric_number,),
  )

  feature_cols = [
      "entry_mode_code",
      "socio_band_code",
      "prior_cgpa",
      "ca_mean",
      "ca_variance",
      "momentum",
      "attendance_rate",
      "clicks_total",
      "duration_total_min",
      "resource_breadth",
      "submission_timeliness",
      "engagement_index",
      "forum_posts_count",
      "login_frequency",
  ]

  student_df.loc[0, "ca_mean"] = new_ca_mean
  student_df.loc[0, "ca_variance"] = new_ca_var

  model = load_model()
  new_prob = float(model.predict_proba(student_df[feature_cols])[:, 1][0])
  new_prob = round(new_prob, 4)

  if new_prob > 0.70:
    new_tier = "High"
  elif new_prob >= 0.50:
    new_tier = "Medium"
  else:
    new_tier = "Low"

  cursor.execute(
      """
        UPDATE students 
        SET ca_mean = ?, ca_variance = ?, risk_probability = ?, risk_level = ?
        WHERE matric_number = ?
    """,
      (new_ca_mean, new_ca_var, new_prob, new_tier, matric_number),
  )

  conn.commit()
  conn.close()


def save_intervention(matric_number, advisor_name, notes, action_taken):
  conn = get_db_connection()
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matric_number TEXT,
            advisor_name TEXT,
            notes TEXT,
            action_taken TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
  cursor.execute(
      """
        INSERT INTO interventions (matric_number, advisor_name, notes, action_taken)
        VALUES (?, ?, ?, ?)
    """,
      (matric_number, advisor_name, notes, action_taken),
  )
  conn.commit()
  conn.close()


model = load_model()
df = load_students()

# --- SESSION STATE MANAGEMENT ---
if "authenticated" not in st.session_state:
  st.session_state["authenticated"] = False
if "user_role" not in st.session_state:
  st.session_state["user_role"] = None
if "user_id" not in st.session_state:
  st.session_state["user_id"] = None

# ==========================================
# PORTAL AUTHENTICATION & ACCOUNT CREATION
# ==========================================
if not st.session_state["authenticated"]:
  st.title("🎓 University Academic Portal")
  st.markdown(
      "Early Warning & Academic Management System. Log in or create an account"
      " below."
  )

  student_tab, staff_tab, register_tab = st.tabs([
      "👨‍🎓 Student Sign In",
      "👨‍🏫 Staff / Lecturer Sign In",
      "➕ Create Account",
  ])

  # 1. STUDENT LOGIN
  with student_tab:
    st.subheader("Student Portal Login")
    with st.form("student_login_form"):
      matric_num = st.text_input(
          "Matriculation Number", placeholder="e.g., MAT-2024-0001"
      )
      student_pass = st.text_input(
          "Password", type="password", value="password123"
      )
      student_submit = st.form_submit_button("Log In as Student")

      if student_submit:
        user = authenticate_user(matric_num, student_pass, "Student")
        if user:
          st.session_state["authenticated"] = True
          st.session_state["user_role"] = "Student"
          st.session_state["user_id"] = user[0]
          st.rerun()
        else:
          st.error("Invalid Matriculation Number or Password.")

  # 2. STAFF LOGIN
  with staff_tab:
    st.subheader("Lecturer & Advisor Portal Login")
    with st.form("staff_login_form"):
      staff_id = st.text_input(
          "Staff ID / Username", placeholder="e.g., admin or LECTURER-01"
      )
      staff_pass = st.text_input("Password", type="password", value="admin123")
      staff_submit = st.form_submit_button("Log In to Dashboard")

      if staff_submit:
        user = authenticate_user(staff_id, staff_pass, "Lecturer/Admin")
        if user:
          st.session_state["authenticated"] = True
          st.session_state["user_role"] = "Lecturer/Admin"
          st.session_state["user_id"] = user[0]
          st.rerun()
        else:
          st.error("Invalid Staff Credentials or Password.")

  # 3. ACCOUNT CREATION
  with register_tab:
    st.subheader("Register New User Account")
    with st.form("registration_form"):
      reg_role = st.selectbox(
          "Select Role", options=["Student", "Lecturer/Admin"]
      )
      reg_name = st.text_input("Full Name", placeholder="e.g., Dr. John Smith")
      reg_username = st.text_input(
          "Matric Number (Students) / Staff ID (Staff)",
          placeholder="e.g., MAT-2024-1001 or LECTURER-02",
      )
      reg_password = st.text_input("Password", type="password")
      reg_confirm_password = st.text_input(
          "Confirm Password", type="password"
      )

      reg_submit = st.form_submit_button("Create Account")

      if reg_submit:
        if not reg_username or not reg_password or not reg_name:
          st.error("Please complete all required fields.")
        elif reg_password != reg_confirm_password:
          st.error("Passwords do not match.")
        else:
          success, msg = register_user(
              reg_username, reg_password, reg_role, reg_name
          )
          if success:
            st.success(msg)
          else:
            st.error(msg)

  st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("👤 Active Session")
st.sidebar.write(f"**User ID:** {st.session_state['user_id']}")
st.sidebar.write(f"**Role:** {st.session_state['user_role']}")

if st.sidebar.button("🔒 Sign Out"):
  st.session_state["authenticated"] = False
  st.session_state["user_role"] = None
  st.session_state["user_id"] = None
  st.rerun()

st.sidebar.markdown("---")

# ==========================================
# VIEW 1: STUDENT PORTAL (FR-17)
# ==========================================
if st.session_state["user_role"] == "Student":
  user_matric = st.session_state["user_id"]
  student_row = df[df["matric_number"] == user_matric]

  if student_row.empty:
    st.warning(
        f"Matric Number '{user_matric}' is registered but has no course records"
        " in the database yet."
    )
    st.stop()

  st.title("🎓 Student Academic Dashboard")
  st.subheader(f"Academic Performance Profile: **{user_matric}**")

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Prior CGPA", f"{student_row['prior_cgpa'].values[0]:.2f} / 4.00")
  col2.metric(
      "Attendance Rate", f"{student_row['attendance_rate'].values[0]:.1f}%"
  )
  col3.metric("CA Mean Score", f"{student_row['ca_mean'].values[0]:.1f}%")
  risk_band = student_row["risk_level"].values[0]
  col4.metric("Academic Risk Band", risk_band)

  st.markdown("---")

  courses_df = load_student_courses(user_matric)
  st.subheader(
      f"📚 Enrolled Course Results ({len(courses_df)} Registered Courses)"
  )
  st.dataframe(courses_df, use_container_width=True)

  st.markdown("---")

  st.subheader("🔍 Personal Performance Analysis & Drivers")
  feature_cols = [
      "entry_mode_code",
      "socio_band_code",
      "prior_cgpa",
      "ca_mean",
      "ca_variance",
      "momentum",
      "attendance_rate",
      "clicks_total",
      "duration_total_min",
      "resource_breadth",
      "submission_timeliness",
      "engagement_index",
      "forum_posts_count",
      "login_frequency",
  ]

  explainer = shap.TreeExplainer(model)
  student_features = student_row[feature_cols]
  shap_values = explainer(student_features)

  contrib_df = pd.DataFrame({
      "Academic Feature": feature_cols,
      "Your Value": student_features.values[0],
      "Impact Score": shap_values.values[0],
  }).sort_values(by="Impact Score", ascending=False)

  st.markdown("##### Key Factors Influencing Your Performance Band:")
  st.dataframe(contrib_df.head(5), use_container_width=True)

  st.markdown("---")
  st.subheader("💡 Recommended Actions")
  if risk_band == "High":
    st.error(
        "⚠️ **High Risk Alert:** Your overall performance flags you for academic"
        " risk. Please contact your Course Advisor immediately."
    )
  elif risk_band == "Medium":
    st.warning(
        "⚡ **Moderate Risk Notice:** Review course assessments under 12/30 and"
        " increase LMS participation to move to Low Risk."
    )
  else:
    st.success(
        "🎉 **On Track:** Excellent performance! Maintain your attendance and"
        " course submissions."
    )

# ==========================================
# VIEW 2: LECTURER & ADVISOR PORTAL (FR-03, FR-16, FR-18, FR-20)
# ==========================================
elif st.session_state["user_role"] == "Lecturer/Admin":
  st.title("🎓 Academic Early Warning & Analytics Dashboard")

  st.sidebar.header("Filter Directory")
  risk_filter = st.sidebar.multiselect(
      "Filter by Risk Level",
      options=["High", "Medium", "Low"],
      default=["High", "Medium", "Low"],
  )

  filtered_df = df[df["risk_level"].isin(risk_filter)].sort_values(
      by="risk_probability", ascending=False
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Total Cohort", len(df))
  col2.metric(
      "High Risk Cohort",
      len(df[df["risk_level"] == "High"]),
      delta_color="inverse",
  )
  col3.metric("Medium Risk Cohort", len(df[df["risk_level"] == "Medium"]))
  col4.metric("Low Risk Cohort", len(df[df["risk_level"] == "Low"]))

  st.markdown("---")

  tab1, tab2, tab3, tab4 = st.tabs([
      "📋 Ordered Class Risk Directory",
      "📝 Input/Update Subject CA Scores",
      "🔍 Student Deep-Dive & SHAP Analytics",
      "📌 Log Advisor Intervention",
  ])

  # TAB 1
  with tab1:
    st.subheader("Class-Wide Risk Directory")
    st.dataframe(
        filtered_df[
            [
                "matric_number",
                "prior_cgpa",
                "attendance_rate",
                "ca_mean",
                "clicks_total",
                "risk_probability",
                "risk_level",
            ]
        ],
        use_container_width=True,
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export At-Risk Cohort Report (CSV)",
        data=csv_data,
        file_name="at_risk_cohort_report.csv",
        mime="text/csv",
    )

  # TAB 2
  with tab2:
    st.subheader("Subject Lecturer Assessment Entry")
    st.markdown(
        "Enter or update Continuous Assessment (CA) scores out of 30 for"
        " enrolled courses. Recalculates risk automatically."
    )

    col_a, col_b = st.columns(2)
    with col_a:
      target_student = st.selectbox(
          "Select Student Matric Number",
          options=df["matric_number"].tolist(),
          key="ca_entry_student",
      )

    student_courses_df = load_student_courses(target_student)

    with col_b:
      target_course = st.selectbox(
          "Select Course",
          options=student_courses_df["Course Code"].tolist(),
          key="ca_entry_course",
      )

    current_score = student_courses_df[
        student_courses_df["Course Code"] == target_course
    ]["CA Score (/30)"].values[0]

    with st.form("update_ca_form"):
      st.write(
          f"Current Score for **{target_course}**: **{current_score:.1f} /"
          " 30**"
      )
      new_score = st.number_input(
          "Enter New CA Score (/30)",
          min_value=0.0,
          max_value=30.0,
          value=float(current_score),
          step=0.5,
      )
      submit_score = st.form_submit_button("Update Score & Recalculate Risk")

      if submit_score:
        update_student_ca_and_repredict(
            target_student, target_course, new_score
        )
        st.success(
            f"Updated CA score for **{target_course}** to **{new_score} / 30**."
            " Risk probabilities updated!"
        )
        st.rerun()

  # TAB 3
  with tab3:
    st.subheader("Student Deep-Dive & Driver Analysis")

    selected_matric = st.selectbox(
        "Select Matric Number for Analysis",
        options=df["matric_number"].tolist(),
    )
    student_row = df[df["matric_number"] == selected_matric]

    if not student_row.empty:
      st.write(
          f"### Risk Classification:"
          f" **{student_row['risk_level'].values[0]}** | Probability:"
          f" **{student_row['risk_probability'].values[0]:.2%}**"
      )

      courses_df = load_student_courses(selected_matric)
      st.subheader(
          f"📚 Course Assessment Breakdown ({len(courses_df)} Courses)"
      )
      st.dataframe(courses_df, use_container_width=True)

      st.markdown("---")
      st.subheader("🤖 Machine Learning Driver Analysis")

      feature_cols = [
          "entry_mode_code",
          "socio_band_code",
          "prior_cgpa",
          "ca_mean",
          "ca_variance",
          "momentum",
          "attendance_rate",
          "clicks_total",
          "duration_total_min",
          "resource_breadth",
          "submission_timeliness",
          "engagement_index",
          "forum_posts_count",
          "login_frequency",
      ]

      explainer = shap.TreeExplainer(model)
      student_features = student_row[feature_cols]
      shap_values = explainer(student_features)

      contrib_df = pd.DataFrame({
          "Feature": feature_cols,
          "Value": student_features.values[0],
          "SHAP Impact Score": shap_values.values[0],
      }).sort_values(by="SHAP Impact Score", ascending=True)

      st.dataframe(contrib_df, use_container_width=True)

  # TAB 4
  with tab4:
    st.subheader("Record Advisor Action / Intervention")

    with st.form("intervention_form"):
      target_matric = st.selectbox(
          "Target Student", options=df["matric_number"].tolist()
      )
      advisor = st.text_input(
          "Advisor / Lecturer Name", value=st.session_state["user_id"]
      )
      action = st.selectbox(
          "Action Taken",
          options=[
              "Academic Counseling Scheduled",
              "Remedial Tutoring Assigned",
              "Parent/Guardian Contacted",
              "Attendance Warning Issued",
          ],
      )
      notes = st.text_area("Detailed Intervention Notes")
      save_btn = st.form_submit_button("Save Intervention Log")

      if save_btn:
        save_intervention(target_matric, advisor, notes, action)
        st.success(
            f"Intervention recorded successfully for **{target_matric}**!"
        )