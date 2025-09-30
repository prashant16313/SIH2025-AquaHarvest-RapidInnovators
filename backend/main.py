from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import joblib
import pandas as pd
import os

# Initialize the FastAPI app
app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://127.0.0.1:5500",  # For VS Code Live Server
    "http://localhost:5500",   # For local testing
    "https://aquaharvestbyrapidinnovators.netlify.app"    # IMPORTANT: Paste your Netlify site URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data.json')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'rwh_model.joblib')
ENCODERS_PATH = os.path.join(SCRIPT_DIR, 'encoders.joblib')


db_data = {}
model = None
encoders = None

# Helper function
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

@app.on_event("startup")
def load_resources():
    global db_data, model, encoders
    print("--- Server is starting up... ---")
    try:
        with open(DB_DATA_PATH, 'r') as f:
            db_data = json.load(f)
        print(f"✅ Database loaded successfully from '{DB_DATA_PATH}'")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load database file. {e}")
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            print(f"✅ AI Model and Encoders loaded successfully.")
        else:
            print(f"⚠️ WARNING: AI Model or encoders not found. Prediction feature will be disabled.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load AI model. {e}")
    print("--- Startup complete. Server is ready. ---")

# --- Pydantic Models ---
class AnalysisInput(BaseModel):
    state: str
    district: str
    rooftopArea: float
    runoffCoefficient: float
    roofTypeText: str
    residents: int
    openSpace: float

def get_structure_recommendation(data, open_space, structure_dimensions):
    geology = data.get('geology', 'unknown').lower()
    pit_area = structure_dimensions.get("pitDimensions", {}).get("area", 5)

    if 'hard-rock' in geology or 'crystalline' in geology:
        if open_space > 500: return {"nameKey": "structure-percolation-tank", "descKey": "structure-percolation-tank-desc", "icon": "pit"}
        elif open_space > 20: return {"nameKey": "structure-recharge-pit", "descKey": "structure-recharge-pit-desc", "icon": "pit"}
        else: return {"nameKey": "structure-storage-tank", "descKey": "structure-storage-tank-desc", "icon": "tank"}
    elif 'alluvium' in geology or 'sedimentary' in geology:
        if open_space > 100: return {"nameKey": "structure-recharge-well", "descKey": "structure-recharge-well-desc", "icon": "borewell"}
        elif open_space > pit_area: return {"nameKey": "structure-recharge-pit", "descKey": "structure-recharge-pit-desc", "icon": "pit"}
        else: return {"nameKey": "structure-storage-tank", "descKey": "structure-storage-tank-desc", "icon": "tank"}
    return {"nameKey": "structure-storage-tank", "descKey": "structure-storage-tank-desc", "icon": "tank"}

def calculate_dimensions(annual_harvest):
    design_volume_liters = (annual_harvest / 365) * 4
    design_volume_m3 = design_volume_liters / 1000
    depth = 3.0
    area = design_volume_m3 / depth if depth > 0 else 0
    side = area**0.5
    return {
        "volume": round(design_volume_m3, 2),
        "pitDimensions": {"text": f"{side:.1f}m L x {side:.1f}m W x {depth}m D", "area": round(area, 2)},
        "tankDimensions": {"text": f"~{round(design_volume_liters)} L capacity"},
        "borewellDimensions": {"text": "Uses borewell. Filter pit (1x1x1m) needed."}
    }

def get_ai_prediction(input_data: AnalysisInput, district_data: dict, recommendation: dict):
    if model is None or encoders is None:
        return 0

    try:
        structure_cleaned = clean_structure_type(recommendation['nameKey'])
        
        # Create DataFrame with all features the new model expects
        input_df = pd.DataFrame([{
            'location.latitude': district_data['coords'][0],
            'location.longitude': district_data['coords'][1],
            'structure_type_cleaned': structure_cleaned,
            'geology': district_data.get('geology', 'unknown'),
            'rainfall': district_data.get('rainfall', 0),
            'groundwaterDepth': district_data.get('groundwaterDepth', 0)
        }])
        
        # Encode categorical features
        label_encoders = {k: v for k, v in encoders.items() if k != 'outcome'}
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                known_classes = set(encoder.classes_)
                input_df[col] = input_df[col].apply(lambda x: x if x in known_classes else encoder.classes_[0])
                input_df[col] = encoder.transform(input_df[col])
        
        probability = model.predict_proba(input_df)
        
        outcome_encoder = encoders['outcome']
        success_class_index = list(outcome_encoder.classes_).index('success')
        success_probability = probability[0][success_class_index]
        
        return int(success_probability * 100)
    except Exception as e:
        print(f"Error during AI prediction: {e}")
        return 0

# --- API Endpoint ---
@app.post("/analyze")
def analyze_data(inputs: AnalysisInput):
    district_data = db_data.get(inputs.state, {}).get(inputs.district)
    if not district_data:
        raise HTTPException(status_code=404, detail="District data not found")

    annual_harvest = round(inputs.rooftopArea * (district_data.get('rainfall', 0) / 1000) * inputs.runoffCoefficient * 1000)
    daily_water_need = inputs.residents * 135
    days_covered = round(annual_harvest / daily_water_need) if daily_water_need > 0 else 0
    all_dimensions = calculate_dimensions(annual_harvest)

    rec_info = get_structure_recommendation(district_data, inputs.openSpace, all_dimensions)
    
    design_volume_liters = (annual_harvest / 365) * 4
    if rec_info['nameKey'] == "structure-storage-tank":
        rec_info['cost'] = design_volume_liters * 6
        rec_info['dimensions'] = all_dimensions['tankDimensions']
    elif rec_info['nameKey'] == "structure-recharge-pit":
        rec_info['cost'] = all_dimensions['volume'] * 4000
        rec_info['dimensions'] = all_dimensions['pitDimensions']
    elif rec_info['nameKey'] == "structure-recharge-well":
        rec_info['cost'] = 50000 
        rec_info['dimensions'] = all_dimensions['borewellDimensions']
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

