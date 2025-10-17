from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
import json
import joblib
import pandas as pd
import os
from datetime import datetime
import pytz
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize the FastAPI app
app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://aquaharvestbyrapidinnovators.netlify.app" # Your Netlify URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading & Firebase Initialization ---
db_data = {}
model = None
encoders = None
db_firestore = None

@app.on_event("startup")
def load_resources():
    global db_data, model, encoders, db_firestore
    print("--- Server is starting up... ---")
    
    # Load local data files
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Adjust path if data.json is in a parent 'data' directory
    DB_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'data.json')
    if not os.path.exists(DB_DATA_PATH):
        DB_DATA_PATH = os.path.join(SCRIPT_DIR, 'data.json') # Fallback to same directory
        
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'rwh_model.joblib')
    ENCODERS_PATH = os.path.join(SCRIPT_DIR, 'encoders.joblib')

    try:
        with open(DB_DATA_PATH, 'r') as f:
            db_data = json.load(f)
        print(f"✅ Database loaded successfully")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load database file from {DB_DATA_PATH}. {e}")
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            print(f"✅ AI Model and Encoders loaded successfully.")
        else:
            print(f"⚠️ WARNING: AI Model or encoders not found.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load AI model. {e}")

    # Initialize Firebase Admin SDK
    try:
        cred_json_str = os.environ.get('FIREBASE_CREDENTIALS')
        if cred_json_str:
            cred_json = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_json)
            firebase_admin.initialize_app(cred)
            db_firestore = firestore.client()
            print("✅ Firebase Admin SDK initialized successfully.")
        else:
            print("⚠️ FIREBASE WARNING: `FIREBASE_CREDENTIALS` env variable not set. Firestore logging is disabled.")
    except Exception as e:
        print(f"❌ FIREBASE ERROR: Could not initialize Firebase Admin SDK. {e}")

    print("--- Startup complete. Server is ready. ---")

# Helper function
def clean_structure_type(st):
    st = str(st).lower()
    if 'percolation' in st: return 'Percolation Tank'
    if 'check dam' in st: return 'Check Dam'
    if 'roof top' in st or 'pavement' in st or 'sump' in st or 'surface' in st: return 'Rooftop Rainwater Harvesting'
    if 'dyke' in st or 'bandhara' in st: return 'Sub-surface Dyke'
    if 'shaft' in st: return 'Recharge Shaft'
    if 'well' in st: return 'Recharge Well'
    if 'trench' in st: return 'Recharge Trench'
    if 'pit' in st: return 'Recharge Pit'
    if 'pond' in st: return 'Farm Pond'
    if 'gully' in st: return 'Gully Plug'
    return 'Other'

# --- Pydantic Models ---
class AnalysisInput(BaseModel):
    state: str
    district: str
    rooftopArea: float
    runoffCoefficient: float
    roofTypeText: str
    residents: int
    openSpace: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # --- YEH INPUTS FRONTEND SE AANE CHAHIYE ---
    userType: Literal['individual', 'community']
    userGoal: Literal['store', 'recharge']
    # --- YEH INPUTS FRONTEND SE FUTURE MEIN ADD KARNE HONGE ---
    has_existing_borewell: Optional[bool] = False # Naya input
    soil_type: Optional[Literal['sandy_loamy', 'clay', 'rocky']] = 'sandy_loamy' # Naya input


# --- UPDATED RECOMMENDATION LOGIC ---
def get_structure_recommendation(inputs: AnalysisInput, district_data: dict, dimensions: dict):
    """
    Smarter recommendation engine based on user type, goal, and site conditions.
    This replaces the old, simplistic logic.
    """
    user_type = inputs.userType
    goal = inputs.userGoal
    open_space = inputs.openSpace
    geology = district_data.get('geology', 'unknown').lower()
    pit_area_needed = dimensions.get("pitDimensions", {}).get("area", 5) 
    design_volume_liters = dimensions.get("tankDimensions", {}).get("volume_liters", 0)

    # --- UPDATED INDIVIDUAL HOME LOGIC ---
    if user_type == 'individual':
        if goal == 'store':
            # Logic ab volume aur cost par based hai
            if design_volume_liters > 7000 and open_space > 10:
                # Badi capacity ke liye Sump behtar hai, agar jagah ho
                return {"nameKey": "structure-rooftop-sump", "descKey": "structure-rooftop-sump-desc", "icon": "tank"}
            else:
                # Chhoti capacity ya kam jagah ke liye Surface Tank
                return {"nameKey": "structure-surface-tank", "descKey": "structure-surface-tank-desc", "icon": "tank"}
        
        elif goal == 'recharge':
            # Naya, step-by-step decision tree logic
            if inputs.has_existing_borewell:
                return {"nameKey": "structure-recharge-well", "descKey": "structure-recharge-well-warning-desc", "icon": "borewell"}
            
            if inputs.soil_type == 'sandy_loamy':
                if open_space > pit_area_needed:
                    return {"nameKey": "structure-recharge-pit", "descKey": "structure-recharge-pit-desc", "icon": "pit"}
                else:
                    # Jagah nahi hai, isliye fallback
                    return {"nameKey": "structure-surface-tank", "descKey": "structure-fallback-storage-desc", "icon": "tank"}
            
            elif inputs.soil_type == 'clay':
                # Mitti aachi nahi hai, isliye fallback
                return {"nameKey": "structure-surface-tank", "descKey": "structure-fallback-clay-soil-desc", "icon": "tank"}
            
            else: # rocky or other
                return {"nameKey": "structure-surface-tank", "descKey": "structure-fallback-storage-desc", "icon": "tank"}

    # --- COMMUNITY / LARGE PROPERTY LOGIC (Placeholder for now) ---
    elif user_type == 'community':
        # Yahan future mein community wala detailed logic aayega
        if open_space > 2000: 
            return {"nameKey": "structure-percolation-tank", "descKey": "structure-percolation-tank-desc", "icon": "pit"}
        elif open_space > 100:
            return {"nameKey": "structure-recharge-trench", "descKey": "structure-recharge-trench-desc", "icon": "pit"}
        else:
            return {"nameKey": "structure-recharge-pit", "descKey": "structure-multi-pit-desc", "icon": "pit"}
            
    # Default fallback
    return {"nameKey": "structure-surface-tank", "descKey": "structure-surface-tank-desc", "icon": "tank"}


def calculate_dimensions(annual_harvest):
    design_volume_liters = (annual_harvest / 365) * 4
    design_volume_m3 = design_volume_liters / 1000
    depth = 3.0
    area = design_volume_m3 / depth if depth > 0 else 0
    side = area**0.5
    return {
        "volume": round(design_volume_m3, 2),
        "pitDimensions": {"text": f"{side:.1f}m L x {side:.1f}m W x {depth}m D", "area": round(area, 2)},
        "tankDimensions": {"text": f"~{round(design_volume_liters)} L capacity", "volume_liters": design_volume_liters},
        "borewellDimensions": {"text": "Uses borewell. Filter pit (1x1x1m) needed."}
    }

def get_ai_prediction(input_data: AnalysisInput, district_data: dict, recommendation: dict):
    if model is None or encoders is None: return 0
    try:
        structure_cleaned = clean_structure_type(recommendation['nameKey'])
        input_df = pd.DataFrame([{
            'location.latitude': district_data['coords'][0],
            'location.longitude': district_data['coords'][1],
            'structure_type_cleaned': structure_cleaned,
            'geology': district_data.get('geology', 'unknown'),
            'rainfall': district_data.get('rainfall', 0),
            'groundwaterDepth': district_data.get('groundwaterDepth', 0)
        }])
        label_encoders = {k: v for k, v in encoders.items() if k != 'outcome'}
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                known_classes = set(encoder.classes_)
                # Handle unseen labels by mapping them to a known class (e.g., the first one)
                input_df[col] = input_df[col].apply(lambda x: x if x in known_classes else encoder.classes_[0])
                input_df[col] = encoder.transform(input_df[col])
        
        probability = model.predict_proba(input_df)
        outcome_encoder = encoders['outcome']
        
        # Check if 'success' class exists
        if 'success' in list(outcome_encoder.classes_):
            success_class_index = list(outcome_encoder.classes_).index('success')
            success_probability = probability[0][success_class_index]
            return int(success_probability * 100)
        else:
            # Fallback if 'success' class is not in the trained model
            return 50 # Return a neutral score

    except Exception as e:
        print(f"Error during AI prediction: {e}")
        return 0 # Return 0 on error

# --- API Endpoint ---
@app.get('/health')
def health_check():
    return {"status": "healthy"}
    
@app.post("/analyze")
def analyze_data(inputs: AnalysisInput):
    IST = pytz.timezone('Asia/Kolkata')
    now_in_ist = datetime.now(IST)
    timestamp_str = now_in_ist.strftime('%d-%m-%Y %H:%M:%S')

    print("="*40)
    print(f"💧 New Analysis Request at: {timestamp_str} (IST)")
    print(f"   Inputs Received: {inputs.dict()}")
    print("="*40)
    
    # Save to Firebase Firestore
    if db_firestore:
        try:
            doc_ref = db_firestore.collection('analysis_requests').document()
            doc_ref.set(inputs.dict())
        except Exception as e:
            print(f"🔥 FIREBASE SAVE FAILED: {e}")

    # Main analysis logic
    district_data = db_data.get(inputs.state, {}).get(inputs.district)
    if not district_data:
        raise HTTPException(status_code=404, detail="District data not found")

    annual_harvest = round(inputs.rooftopArea * (district_data.get('rainfall', 0) / 1000) * inputs.runoffCoefficient * 1000)
    daily_water_need = inputs.residents * 135
    days_covered = round(annual_harvest / daily_water_need) if daily_water_need > 0 else 0
    all_dimensions = calculate_dimensions(annual_harvest)

    # Use the NEW recommendation function
    rec_info = get_structure_recommendation(inputs, district_data, all_dimensions)
    
    # Cost and Dimension assignment based on recommendation
    if rec_info['nameKey'] in ["structure-storage-tank", "structure-surface-tank", "structure-rooftop-sump"]:
        rec_info['cost'] = all_dimensions['tankDimensions']['volume_liters'] * 6 # Approx cost
        rec_info['dimensions'] = all_dimensions['tankDimensions']
    elif rec_info['nameKey'] == "structure-recharge-pit":
        rec_info['cost'] = all_dimensions['volume'] * 4000
        rec_info['dimensions'] = all_dimensions['pitDimensions']
    elif rec_info['nameKey'] == "structure-recharge-well":
        rec_info['cost'] = 50000 
        rec_info['dimensions'] = all_dimensions['borewellDimensions']
    elif rec_info['nameKey'] == "structure-recharge-trench":
        rec_info['cost'] = all_dimensions['volume'] * 3000
        rec_info['dimensions'] = all_dimensions['pitDimensions'] # Approx
    elif rec_info['nameKey'] == "structure-percolation-tank":
        rec_info['cost'] = 100000 # Placeholder
        rec_info['dimensions'] = {"text": "Varies by specific design."}
    else:
        rec_info['cost'] = 30000
        rec_info['dimensions'] = {"text": "Varies by specific design."}
    
    rec_info['cost'] = round(rec_info['cost'])
    rec_info['annualSavings'] = round((annual_harvest / 1000) * (district_data.get('municipalWaterRate', 10)))
    
    payback_period = rec_info['cost'] / rec_info['annualSavings'] if rec_info['annualSavings'] > 0 else float('inf')
    if payback_period <= 7:
        feasibility = { 'ratingKey': 'feasibility-highly', 'textKey': 'feasibility-highly-text', 'color': 'green' }
    elif payback_period <= 15:
        feasibility = { 'ratingKey': 'feasibility-moderately', 'textKey': 'feasibility-moderately-text', 'color': 'yellow' }
    else:
        feasibility = { 'ratingKey': 'feasibility-low', 'textKey': 'feasibility-low-text', 'color': 'orange' }

    ai_prediction_score = get_ai_prediction(inputs, district_data, rec_info)

    return {
        "inputs": inputs.dict(),
        "data": district_data,
        "annualHarvest": annual_harvest,
        "daysCovered": days_covered,
        "recommendation": rec_info,
        "feasibility": feasibility,
        "aiSuccessPrediction": ai_prediction_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

