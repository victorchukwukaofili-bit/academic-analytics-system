import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

n_students = 1000
matric_numbers = [f"MAT-2024-{i+1:04d}" for i in range(n_students)]

course_catalog = [
    ("MAT 101", "General Mathematics I"),
    ("PHY 101", "General Physics I"),
    ("CSC 101", "Introduction to Computer Science"),
    ("GST 101", "Use of English"),
    ("CHM 101", "General Chemistry I"),
    ("STA 101", "Introductory Statistics"),
    ("ENG 101", "Engineering Mechanics"),
    ("BIO 101", "General Biology I"),
    ("GST 102", "Nigerian Peoples and Culture"),
    ("PHY 102", "Practical Physics I"),
    ("CHM 102", "Practical Chemistry I"),
    ("CSC 102", "Introduction to Problem Solving"),
    ("MAT 102", "General Mathematics II"),
]

student_records = []
course_records = []

for i, matric in enumerate(matric_numbers):
  entry_mode = np.random.choice([0, 1, 2], p=[0.6, 0.25, 0.15])
  socio_band = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
  prior_cgpa = round(float(np.clip(np.random.normal(2.8, 0.7), 1.0, 4.0)), 2)

  n_courses = random.randint(1, 11)
  selected_courses = random.sample(course_catalog, n_courses)

  scores = []
  for code, title in selected_courses:
    ca_score = round(
        float(np.clip(prior_cgpa * 6 + np.random.normal(0, 4.0), 0, 30)), 1
    )
    scores.append(ca_score)
    course_records.append({
        "matric_number": matric,
        "course_code": code,
        "course_title": title,
        "ca_score_30": ca_score,
    })

  ca_mean = round(float((np.mean(scores) / 30) * 100), 2)
  ca_variance = round(float(np.var(scores)), 2) if len(scores) > 1 else 0.0
  momentum = round(float(np.random.normal(0, 1)), 2)

  attendance_rate = round(
      float(np.clip(np.random.beta(3, 2) * 100, 0, 100)), 2
  )
  clicks_total = int(np.random.poisson(lam=250))
  duration_total_min = round(float(clicks_total * np.random.uniform(1.5, 3.5)), 2)
  resource_breadth = int(np.random.randint(1, 15))
  submission_timeliness = round(float(np.random.normal(0, 24)), 2)
  engagement_index = round(
      float(
          clicks_total * 0.4
          + attendance_rate * 0.4
          + (15 - resource_breadth) * 0.2
      ),
      2,
  )
  forum_posts_count = int(np.random.poisson(lam=5))
  login_frequency = int(np.random.poisson(lam=20))

  student_records.append({
      "matric_number": matric,
      "entry_mode_code": entry_mode,
      "socio_band_code": socio_band,
      "prior_cgpa": prior_cgpa,
      "ca_mean": ca_mean,
      "ca_variance": ca_variance,
      "momentum": momentum,
      "attendance_rate": attendance_rate,
      "clicks_total": clicks_total,
      "duration_total_min": duration_total_min,
      "resource_breadth": resource_breadth,
      "submission_timeliness": submission_timeliness,
      "engagement_index": engagement_index,
      "forum_posts_count": forum_posts_count,
      "login_frequency": login_frequency,
  })

df_students = pd.DataFrame(student_records)

# Create a clear risk formula and split at median so exactly ~50% are at risk
risk_score = (
    - 1.5 * df_students["prior_cgpa"]
    - 0.05 * df_students["ca_mean"]
    - 0.03 * df_students["attendance_rate"]
    - 0.003 * df_students["clicks_total"]
    + 0.02 * df_students["ca_variance"]
    + np.random.normal(0, 1, n_students)
)

df_students["is_at_risk"] = (risk_score > np.median(risk_score)).astype(int)

# Save datasets
df_students.to_csv("student_data.csv", index=False)
pd.DataFrame(course_records).to_csv("student_courses.csv", index=False)

print("Dataset rebuilt with a balanced risk distribution!")