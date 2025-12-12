import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 200

if n_samples % 2 != 0:
    n_samples += 1

data = {
    'age': np.random.randint(18, 85, n_samples),
    'gender': np.random.choice([0, 1], n_samples),
    
    'body_temperature': np.random.uniform(36.5, 39.5, n_samples),
    'heart_rate': np.random.randint(60, 130, n_samples),
    'respiratory_rate': np.random.randint(12, 35, n_samples),
    'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
    'oxygen_saturation': np.random.uniform(85, 100, n_samples),
    
    'white_blood_cell_count': np.random.uniform(4.0, 15.0, n_samples),
    'c_reactive_protein': np.random.uniform(0, 150, n_samples),
    'lymphocyte_count': np.random.uniform(0.8, 4.0, n_samples),
    
    'cough': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'fever': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'fatigue': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
    'shortness_of_breath': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    'chest_pain': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    
    'has_diabetes': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'has_hypertension': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'smoking_status': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
}

df = pd.DataFrame(data)

infection_score = (
    (df['body_temperature'] > 38.0).astype(int) * 0.3 +
    (df['c_reactive_protein'] > 50).astype(int) * 0.25 +
    (df['white_blood_cell_count'] > 10).astype(int) * 0.2 +
    (df['oxygen_saturation'] < 95).astype(int) * 0.25 +
    df['cough'] * 0.15 +
    df['fever'] * 0.2 +
    df['shortness_of_breath'] * 0.2 +
    (df['age'] > 60).astype(int) * 0.15 +
    df['has_diabetes'] * 0.1 +
    np.random.uniform(-0.2, 0.2, n_samples)
)

sorted_indices = np.argsort(infection_score)

n_per_class = n_samples // 2
df['infected'] = 0
df.loc[sorted_indices[-n_per_class:], 'infected'] = 1

shuffled_indices = np.random.permutation(df.index)
df = df.loc[shuffled_indices].reset_index(drop=True)

print("Clinical Dataset - Respiratory Infection Prediction")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"\nTarget Distribution:")
print(df['infected'].value_counts())
print(f"\nNot Infected: {(df['infected'] == 0).sum()} ({(df['infected'] == 0).sum()/len(df)*100:.1f}%)")
print(f"Infected: {(df['infected'] == 1).sum()} ({(df['infected'] == 1).sum()/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("Feature Descriptions:")
print("=" * 60)
print("""
Demographics:
  - age: Patient age (18-85 years)
  - gender: 0=Female, 1=Male

Vital Signs:
  - body_temperature: Body temperature in Celsius (36.5-39.5°C)
  - heart_rate: Heart rate in beats per minute (60-130 bpm)
  - respiratory_rate: Breathing rate (12-35 breaths/min)
  - blood_pressure_systolic: Systolic blood pressure (90-180 mmHg)
  - oxygen_saturation: Blood oxygen level (85-100%)

Laboratory Results:
  - white_blood_cell_count: WBC count (4.0-15.0 x10^9/L)
  - c_reactive_protein: CRP inflammation marker (0-150 mg/L)
  - lymphocyte_count: Lymphocyte count (0.8-4.0 x10^9/L)

Symptoms (0=No, 1=Yes):
  - cough, fever, fatigue, shortness_of_breath, chest_pain

Medical History:
  - has_diabetes: 0=No, 1=Yes
  - has_hypertension: 0=No, 1=Yes
  - smoking_status: 0=Never, 1=Former, 2=Current

Target:
  - infected: 0=Not Infected, 1=Infected
""")

print("=" * 60)
print("\nFirst 10 samples:")
print(df.head(10))

print("\n" + "=" * 60)
print("Statistical Summary:")
print(df.describe())

csv_filename = 'clclinical_respiratory_infection_dataset.csv'
df.to_csv(csv_filename, index=False)
print(f"\n{'='*60}")
print(f"Dataset saved to: {csv_filename}")
print(f"{'='*60}")
