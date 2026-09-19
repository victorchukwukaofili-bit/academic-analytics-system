import sqlite3
import joblib
import pandas as pd

# 1. Load generated datasets
df_students = pd.read_csv("student_data.csv")
df_courses = pd.read_csv("student_courses.csv")

# 2. Load model and compute risk predictions
model = joblib.load("model.pkl")

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

probs = model.predict_proba(df_students[feature_cols])[:, 1]
df_students["risk_probability"] = probs.round(4)


def get_risk_level(p):
  if p > 0.70:
    return "High"
  elif p >= 0.50:
    return "Medium"
  else:
    return "Low"


df_students["risk_level"] = df_students["risk_probability"].apply(
    get_risk_level
)

# 3. Connect to SQLite and sync tables
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Save core data tables
df_students.to_sql("students", conn, if_exists="replace", index=False)
df_courses.to_sql("student_courses", conn, if_exists="replace", index=False)

# Create User Authentication Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        full_name TEXT
    )
""")

# Seed default admin/lecturer account
cursor.execute("""
    INSERT OR IGNORE INTO users (username, password, role, full_name)
    VALUES ('admin', 'admin123', 'Lecturer/Admin', 'System Administrator')
""")

# Seed student accounts from dataset
for _, row in df_students.iterrows():
  cursor.execute(
      """
        INSERT OR IGNORE INTO users (username, password, role, full_name)
        VALUES (?, 'password123', 'Student', ?)
    """,
      (row["matric_number"], row["matric_number"]),
  )

conn.commit()
conn.close()

print("Database updated with users, students, and course tables!")