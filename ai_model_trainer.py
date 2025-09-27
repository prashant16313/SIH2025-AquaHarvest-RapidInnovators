import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- File Paths ---
# Ensure this script is run from the project's root directory.
TRAINING_DATA_PATH = 'ai_training_data.json'
DB_DATA_PATH = 'data.json' # Main data file with rainfall, etc.
MODEL_SAVE_PATH = 'backend/rwh_model.joblib'
ENCODERS_SAVE_PATH = 'backend/encoders.joblib'

def clean_structure_type(st):
    """Standardizes the structure type string."""
    st = str(st).lower()
    if 'percolation' in st: return 'Percolation Tank'
    if 'check dam' in st: return 'Check Dam'
    if 'roof top' in st or 'pavement' in st: return 'Rooftop Rainwater Harvesting'
    if 'dyke' in st or 'bandhara' in st: return 'Sub-surface Dyke'
    if 'shaft' in st: return 'Recharge Shaft'
    if 'well' in st: return 'Recharge Well'
    if 'trench' in st: return 'Recharge Trench'
    if 'pond' in st: return 'Farm Pond'
    if 'gully' in st: return 'Gully Plug'
    return 'Other'

def merge_data(training_df, db_data):
    """Merges training data with the main database to add features."""
    additional_features = []
    for index, row in training_df.iterrows():
        state = row.get('state')
        district = row.get('district')
        features = {
            'rainfall': None,
            'groundwaterDepth': None
        }
        if state in db_data and district in db_data[state]:
            district_info = db_data[state][district]
            features['rainfall'] = district_info.get('rainfall')
            features['groundwaterDepth'] = district_info.get('groundwaterDepth')
        additional_features.append(features)
    
    features_df = pd.DataFrame(additional_features)
    return pd.concat([training_df, features_df], axis=1)

def train_model():
    """
    Loads, merges, tunes, and trains a more robust AI model.
    """
    print("--- Starting Improved AI Model Training Process ---")

    # 1. Load Data
    try:
        with open(TRAINING_DATA_PATH, 'r') as f:
            training_data = json.load(f)
        with open(DB_DATA_PATH, 'r') as f:
            db_data = json.load(f)
        print(f"✅ Step 1: Successfully loaded {len(training_data)} training records and main database.")
    except Exception as e:
        print(f"❌ ERROR in Step 1: Could not load data files. Details: {e}")
        return

    df = pd.json_normalize(training_data)

    # 2. Feature Engineering: Merge data to add new features
    df = merge_data(df, db_data)
    df.dropna(subset=['rainfall', 'groundwaterDepth'], inplace=True) # Drop rows where we couldn't find features
    
    df['structure_type_cleaned'] = df['structure_type'].apply(clean_structure_type)
    
    if 'geology' not in df.columns: df['geology'] = 'unknown'
    else: df['geology'].fillna('unknown', inplace=True)

    # UPDATED: Add new features to the model
    features = ['location.latitude', 'location.longitude', 'structure_type_cleaned', 'geology', 'rainfall', 'groundwaterDepth']
    target = 'outcome'
    X = df[features]
    y = df[target]

    print("\nSample of features being used for training (with new features):")
    print(X.head())
    
    # 3. Encode Categorical Data
    encoders = {}
    X = X.copy()
    for col in ['structure_type_cleaned', 'geology']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    le_outcome = LabelEncoder()
    y_encoded = le_outcome.fit_transform(y)
    encoders[target] = le_outcome
    print("\n✅ Step 2 & 3: Feature Engineering and Encoding complete.")
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    print(f"✅ Step 4: Data split into {len(X_train)} training and {len(X_test)} testing records.")

    # 5. Hyperparameter Tuning with GridSearchCV
    print("\n⏳ Step 5: Finding best model settings with GridSearchCV...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Best parameters found: {grid_search.best_params_}")

    # 6. Evaluate the Best Model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📈 Step 6: Best Model Evaluation:")
    print(f"   - Accuracy on test data: {accuracy * 100:.2f}%")
    print("   - Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le_outcome.classes_, zero_division=0)
    for line in report.split('\n'):
        print(f"     {line}")

    # 7. Save the best model and encoders
    joblib.dump(best_model, MODEL_SAVE_PATH)
    joblib.dump(encoders, ENCODERS_SAVE_PATH)
    print(f"\n✅ Step 7: Best model saved to '{MODEL_SAVE_PATH}'")
    print(f"✅ Encoders saved to '{ENCODERS_SAVE_PATH}'")
    print("\n--- AI Training Complete! The new model is more powerful. ---")

if __name__ == "__main__":
    if not os.path.exists('backend'):
        os.makedirs('backend')
    train_model()
