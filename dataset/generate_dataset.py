"""
Balanced Dataset Generator for Smart Crop Advisory System
=========================================================
Generates a balanced ML-ready crop recommendation dataset using:
  - Season × Region → Crop mappings
  - Crop-specific agronomic profiles
  - Balanced per-crop sampling
  - Preferred soil bias
  - Small realistic variation / hard cases

Output:
    training_dataset.csv
"""

import csv
import random
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
SOIL_TYPES = ["Sandy", "Clay", "Loamy", "Silt", "Peaty", "Chalky", "Saline"]

# Keep exactly same season-region mapping as your app
HIGH_DEMAND_CROPS = {
    "Kuruvai (Jun – Sep) – TN Early Kharif": {
        "Tamil Nadu – Cauvery Delta": ["Short-duration Rice", "Sesame", "Groundnut"],
        "Tamil Nadu – Coastal Tamil Nadu": ["Rice", "Blackgram", "Sesame"],
        "Tamil Nadu – Southern Dry Zone": ["Pearl Millet", "Groundnut", "Sesame"],
        "Tamil Nadu – Nilgiris / High Altitude Zone": ["Potato", "Carrot", "Cabbage"],
    },
    "Samba (Aug – Jan) – TN Main Season": {
        "Tamil Nadu – Cauvery Delta": ["Samba Rice", "Sugarcane", "Banana"],
        "Tamil Nadu – Coastal Tamil Nadu": ["Rice", "Sugarcane", "Groundnut"],
        "Tamil Nadu – Southern Dry Zone": ["Sorghum", "Cotton", "Castor"],
        "Tamil Nadu – Nilgiris / High Altitude Zone": ["Potato", "Carrot", "Cabbage"],
    },
    "Navarai (Jan – Mar) – TN Summer Crop": {
        "Tamil Nadu – Cauvery Delta": ["Rice", "Blackgram", "Greengram"],
        "Tamil Nadu – Coastal Tamil Nadu": ["Rice", "Groundnut", "Sesame"],
        "Tamil Nadu – Southern Dry Zone": ["Pearl Millet", "Groundnut", "Sunflower"],
        "Tamil Nadu – Nilgiris / High Altitude Zone": ["Potato", "Carrot", "Beans"],
    },
    "Kharif / SW Monsoon (Jun – Oct) – AP / Telangana / Karnataka": {
        "Andhra Pradesh – Godavari Delta": ["Rice", "Maize", "Sugarcane"],
        "Andhra Pradesh – Krishna Delta": ["Rice", "Maize", "Banana"],
        "Andhra Pradesh – Coastal Andhra": ["Groundnut", "Cotton", "Rice"],
        "Andhra Pradesh – Rayalaseema": ["Groundnut", "Redgram", "Sunflower"],
        "Telangana – Northern Telangana Dry Zone": ["Cotton", "Maize", "Soybean"],
        "Telangana – Eastern Ghats Upland": ["Maize", "Redgram", "Millets"],
        "Karnataka – Deccan Plateau": ["Maize", "Groundnut", "Turmeric"],
        "Karnataka – Krishna–Tungabhadra Basin": ["Cotton", "Maize", "Chilli"],
        "Karnataka – Northern Dry Zone": ["Sorghum", "Cotton", "Groundnut"],
        "Karnataka – Malnad (Western Ghats)": ["Rice", "Arecanut", "Pepper"],
    },
    "Rabi / Winter (Nov – Feb) – AP / Telangana / Karnataka": {
        "Andhra Pradesh – Godavari Delta": ["Rabi Rice", "Blackgram", "Maize"],
        "Andhra Pradesh – Krishna Delta": ["Rice", "Blackgram", "Chilli"],
        "Andhra Pradesh – Coastal Andhra": ["Groundnut", "Blackgram", "Sunflower"],
        "Andhra Pradesh – Rayalaseema": ["Chickpea", "Groundnut", "Sunflower"],
        "Telangana – Northern Telangana Dry Zone": ["Maize", "Chickpea", "Safflower"],
        "Telangana – Eastern Ghats Upland": ["Millets", "Pigeonpea", "Chickpea"],
        "Karnataka – Deccan Plateau": ["Wheat", "Chickpea", "Sunflower"],
        "Karnataka – Krishna–Tungabhadra Basin": ["Sorghum", "Chickpea", "Onion"],
        "Karnataka – Northern Dry Zone": ["Wheat", "Safflower", "Chickpea"],
        "Karnataka – Malnad (Western Ghats)": ["Vegetables", "Coffee", "Pepper"],
    },
    "Summer / Zaid (Mar – May) – AP / Telangana / Karnataka": {
        "Andhra Pradesh – Godavari Delta": ["Watermelon", "Cucumber", "Rice"],
        "Andhra Pradesh – Krishna Delta": ["Muskmelon", "Groundnut", "Vegetables"],
        "Andhra Pradesh – Coastal Andhra": ["Watermelon", "Groundnut", "Sesame"],
        "Andhra Pradesh – Rayalaseema": ["Sunflower", "Groundnut", "Sesame"],
        "Telangana – Northern Telangana Dry Zone": ["Greengram", "Watermelon", "Sesame"],
        "Telangana – Eastern Ghats Upland": ["Millets", "Greengram", "Vegetables"],
        "Karnataka – Deccan Plateau": ["Greengram", "Watermelon", "Cucumber"],
        "Karnataka – Krishna–Tungabhadra Basin": ["Greengram", "Sunflower", "Vegetables"],
        "Karnataka – Northern Dry Zone": ["Greengram", "Watermelon", "Sesame"],
        "Karnataka – Malnad (Western Ghats)": ["Vegetables", "Banana", "Turmeric"],
    },
    "Virippu / SW Monsoon (Jun – Aug) – Kerala": {
        "Kerala – Palakkad Gap": ["Rice", "Banana", "Vegetables"],
        "Kerala – Western Ghats (High Rainfall Zone)": ["Rubber", "Cardamom", "Pepper"],
        "Kerala – Coastal Kerala": ["Coconut", "Tapioca", "Banana"],
    },
    "Mundakan / NE Monsoon (Sep – Nov) – Kerala": {
        "Kerala – Palakkad Gap": ["Rice", "Cowpea", "Sesame"],
        "Kerala – Western Ghats (High Rainfall Zone)": ["Ginger", "Turmeric", "Pepper"],
        "Kerala – Coastal Kerala": ["Tapioca", "Banana", "Vegetables"],
    },
    "Puncha / Winter Paddy (Dec – Feb) – Kerala": {
        "Kerala – Palakkad Gap": ["Puncha Rice", "Vegetables", "Watermelon"],
        "Kerala – Western Ghats (High Rainfall Zone)": ["Coffee", "Cardamom", "Tea"],
        "Kerala – Coastal Kerala": ["Coconut", "Vegetables", "Banana"],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# CROP PROFILES
# ──────────────────────────────────────────────────────────────────────────────
CROP_PROFILES = {
    "Rice":               dict(nitrogen=(80,140), phosphorus=(40,80), potassium=(40,80),  ph=(5.5,7.0), moisture=(60,90), temperature=(22,35), humidity=(60,90), rainfall=(1200,2500), preferred_soils=["Clay","Loamy","Silt"]),
    "Short-duration Rice":dict(nitrogen=(70,130), phosphorus=(35,75), potassium=(35,75),  ph=(5.5,7.0), moisture=(60,90), temperature=(22,35), humidity=(60,90), rainfall=(1000,2200), preferred_soils=["Clay","Loamy","Silt"]),
    "Samba Rice":         dict(nitrogen=(90,140), phosphorus=(40,80), potassium=(40,80),  ph=(5.5,7.0), moisture=(65,90), temperature=(22,35), humidity=(65,90), rainfall=(1200,2500), preferred_soils=["Clay","Loamy"]),
    "Rabi Rice":          dict(nitrogen=(80,130), phosphorus=(40,80), potassium=(40,75),  ph=(5.5,7.0), moisture=(55,80), temperature=(18,30), humidity=(55,80), rainfall=(800,1800),  preferred_soils=["Clay","Loamy","Silt"]),
    "Puncha Rice":        dict(nitrogen=(80,130), phosphorus=(35,75), potassium=(35,75),  ph=(5.5,7.0), moisture=(60,85), temperature=(18,28), humidity=(60,85), rainfall=(800,1800),  preferred_soils=["Clay","Loamy"]),
    "Wheat":              dict(nitrogen=(60,120), phosphorus=(40,80), potassium=(40,80),  ph=(6.0,7.5), moisture=(40,65), temperature=(10,25), humidity=(40,70), rainfall=(400,1000),  preferred_soils=["Loamy","Clay","Silt"]),
    "Maize":              dict(nitrogen=(60,120), phosphorus=(40,80), potassium=(60,120), ph=(5.8,7.5), moisture=(40,70), temperature=(20,35), humidity=(50,80), rainfall=(600,1200),  preferred_soils=["Loamy","Sandy","Silt"]),
    "Sorghum":            dict(nitrogen=(40,100), phosphorus=(20,60), potassium=(40,80),  ph=(5.5,8.0), moisture=(30,65), temperature=(25,40), humidity=(30,70), rainfall=(400,1000),  preferred_soils=["Loamy","Sandy","Clay"]),
    "Pearl Millet":       dict(nitrogen=(40,90),  phosphorus=(20,60), potassium=(30,70),  ph=(6.0,8.0), moisture=(25,60), temperature=(28,42), humidity=(25,65), rainfall=(300,900),   preferred_soils=["Sandy","Loamy"]),
    "Millets":            dict(nitrogen=(35,85),  phosphorus=(20,55), potassium=(30,70),  ph=(5.8,8.0), moisture=(25,60), temperature=(24,40), humidity=(30,70), rainfall=(300,900),   preferred_soils=["Sandy","Loamy"]),
    "Cotton":             dict(nitrogen=(60,100), phosphorus=(40,80), potassium=(60,120), ph=(6.0,8.0), moisture=(40,70), temperature=(25,40), humidity=(40,70), rainfall=(600,1200),  preferred_soils=["Clay","Loamy","Silt"]),
    "Sugarcane":          dict(nitrogen=(100,150),phosphorus=(50,90), potassium=(100,200),ph=(6.0,7.5), moisture=(60,90), temperature=(24,38), humidity=(50,80), rainfall=(1200,2000), preferred_soils=["Loamy","Clay","Silt"]),
    "Banana":             dict(nitrogen=(100,150),phosphorus=(50,80), potassium=(150,250),ph=(5.5,7.5), moisture=(60,85), temperature=(22,38), humidity=(60,85), rainfall=(1200,2500), preferred_soils=["Loamy","Silt"]),
    "Coconut":            dict(nitrogen=(60,100), phosphorus=(30,60), potassium=(100,200),ph=(5.5,8.0), moisture=(50,80), temperature=(22,38), humidity=(60,90), rainfall=(1200,2500), preferred_soils=["Sandy","Loamy"]),
    "Arecanut":           dict(nitrogen=(60,100), phosphorus=(30,60), potassium=(80,150), ph=(5.5,7.5), moisture=(60,85), temperature=(22,38), humidity=(65,90), rainfall=(1500,3000), preferred_soils=["Loamy","Peaty"]),
    "Groundnut":          dict(nitrogen=(20,50),  phosphorus=(50,90), potassium=(40,80),  ph=(6.0,7.5), moisture=(40,65), temperature=(25,35), humidity=(40,70), rainfall=(500,1000),  preferred_soils=["Sandy","Loamy"]),
    "Sesame":             dict(nitrogen=(20,60),  phosphorus=(20,60), potassium=(30,70),  ph=(5.5,8.0), moisture=(25,55), temperature=(28,40), humidity=(25,60), rainfall=(300,800),   preferred_soils=["Sandy","Loamy"]),
    "Sunflower":          dict(nitrogen=(60,100), phosphorus=(40,80), potassium=(40,80),  ph=(6.0,7.5), moisture=(35,65), temperature=(20,35), humidity=(35,65), rainfall=(500,1000),  preferred_soils=["Loamy","Sandy","Clay"]),
    "Castor":             dict(nitrogen=(40,80),  phosphorus=(30,60), potassium=(40,80),  ph=(5.5,8.0), moisture=(30,60), temperature=(25,40), humidity=(30,65), rainfall=(400,900),   preferred_soils=["Sandy","Loamy"]),
    "Safflower":          dict(nitrogen=(40,80),  phosphorus=(30,60), potassium=(30,60),  ph=(6.0,8.0), moisture=(25,55), temperature=(20,35), humidity=(25,60), rainfall=(300,750),   preferred_soils=["Loamy","Clay","Sandy"]),
    "Soybean":            dict(nitrogen=(20,50),  phosphorus=(50,90), potassium=(60,100), ph=(6.0,7.5), moisture=(45,75), temperature=(20,35), humidity=(50,80), rainfall=(600,1200),  preferred_soils=["Loamy","Clay","Silt"]),
    "Chickpea":           dict(nitrogen=(20,50),  phosphorus=(50,90), potassium=(40,80),  ph=(6.0,8.0), moisture=(30,60), temperature=(15,28), humidity=(30,65), rainfall=(350,800),   preferred_soils=["Loamy","Sandy","Clay"]),
    "Blackgram":          dict(nitrogen=(20,50),  phosphorus=(40,80), potassium=(30,70),  ph=(6.0,7.5), moisture=(40,65), temperature=(25,35), humidity=(50,80), rainfall=(500,1000),  preferred_soils=["Loamy","Sandy"]),
    "Greengram":          dict(nitrogen=(20,50),  phosphorus=(40,80), potassium=(30,70),  ph=(6.0,7.5), moisture=(35,65), temperature=(25,38), humidity=(45,75), rainfall=(400,900),   preferred_soils=["Sandy","Loamy"]),
    "Redgram":            dict(nitrogen=(20,50),  phosphorus=(40,80), potassium=(30,70),  ph=(6.0,7.5), moisture=(35,65), temperature=(25,38), humidity=(40,75), rainfall=(600,1200),  preferred_soils=["Loamy","Sandy","Clay"]),
    "Pigeonpea":          dict(nitrogen=(20,50),  phosphorus=(40,80), potassium=(30,70),  ph=(6.0,7.5), moisture=(35,65), temperature=(25,38), humidity=(40,75), rainfall=(600,1200),  preferred_soils=["Loamy","Sandy"]),
    "Cowpea":             dict(nitrogen=(20,50),  phosphorus=(30,70), potassium=(30,70),  ph=(5.5,7.5), moisture=(35,65), temperature=(25,38), humidity=(45,75), rainfall=(400,900),   preferred_soils=["Sandy","Loamy"]),
    "Potato":             dict(nitrogen=(80,140), phosphorus=(60,100),potassium=(100,200),ph=(5.0,7.0), moisture=(55,80), temperature=(10,22), humidity=(60,85), rainfall=(600,1200),  preferred_soils=["Loamy","Sandy","Silt"]),
    "Tomato":             dict(nitrogen=(80,130), phosphorus=(50,90), potassium=(80,150), ph=(6.0,7.0), moisture=(55,80), temperature=(18,32), humidity=(50,80), rainfall=(600,1200),  preferred_soils=["Loamy","Sandy"]),
    "Onion":              dict(nitrogen=(60,100), phosphorus=(40,80), potassium=(60,120), ph=(6.0,7.5), moisture=(50,75), temperature=(15,30), humidity=(40,70), rainfall=(500,1000),  preferred_soils=["Loamy","Sandy","Silt"]),
    "Carrot":             dict(nitrogen=(50,90),  phosphorus=(40,80), potassium=(60,120), ph=(6.0,7.0), moisture=(50,75), temperature=(10,22), humidity=(50,80), rainfall=(500,1000),  preferred_soils=["Sandy","Loamy"]),
    "Cabbage":            dict(nitrogen=(80,130), phosphorus=(40,80), potassium=(80,140), ph=(6.0,7.5), moisture=(55,80), temperature=(10,22), humidity=(55,80), rainfall=(500,1000),  preferred_soils=["Loamy","Clay","Silt"]),
    "Tapioca":            dict(nitrogen=(60,100), phosphorus=(30,70), potassium=(80,150), ph=(5.5,7.5), moisture=(50,80), temperature=(24,38), humidity=(60,85), rainfall=(1000,2000), preferred_soils=["Sandy","Loamy"]),
    "Watermelon":         dict(nitrogen=(50,90),  phosphorus=(30,70), potassium=(60,120), ph=(6.0,7.5), moisture=(40,70), temperature=(25,38), humidity=(40,70), rainfall=(400,800),   preferred_soils=["Sandy","Loamy"]),
    "Muskmelon":          dict(nitrogen=(50,90),  phosphorus=(30,70), potassium=(60,120), ph=(6.0,7.5), moisture=(40,70), temperature=(25,38), humidity=(40,70), rainfall=(400,800),   preferred_soils=["Sandy","Loamy"]),
    "Cucumber":           dict(nitrogen=(60,100), phosphorus=(40,80), potassium=(60,120), ph=(6.0,7.5), moisture=(50,80), temperature=(22,35), humidity=(55,80), rainfall=(500,1000),  preferred_soils=["Sandy","Loamy","Silt"]),
    "Beans":              dict(nitrogen=(30,70),  phosphorus=(40,80), potassium=(40,80),  ph=(6.0,7.5), moisture=(50,75), temperature=(15,28), humidity=(55,80), rainfall=(600,1200),  preferred_soils=["Loamy","Sandy"]),
    "Vegetables":         dict(nitrogen=(50,100), phosphorus=(40,80), potassium=(50,100), ph=(5.8,7.5), moisture=(50,80), temperature=(15,35), humidity=(50,80), rainfall=(500,1500),  preferred_soils=["Loamy","Sandy","Silt"]),
    "Chilli":             dict(nitrogen=(60,100), phosphorus=(40,80), potassium=(60,120), ph=(6.0,7.5), moisture=(40,70), temperature=(20,35), humidity=(40,75), rainfall=(500,1200), preferred_soils=["Loamy","Sandy"]),
    "Tea":                dict(nitrogen=(100,150),phosphorus=(30,60), potassium=(60,120), ph=(4.5,6.0), moisture=(65,90), temperature=(12,28), humidity=(70,95), rainfall=(1500,3000), preferred_soils=["Loamy","Peaty"]),
    "Coffee":             dict(nitrogen=(80,130), phosphorus=(30,70), potassium=(80,140), ph=(5.5,6.5), moisture=(60,85), temperature=(18,28), humidity=(65,90), rainfall=(1500,2500), preferred_soils=["Loamy","Peaty"]),
    "Rubber":             dict(nitrogen=(60,100), phosphorus=(30,60), potassium=(60,120), ph=(4.5,6.5), moisture=(65,90), temperature=(22,35), humidity=(70,95), rainfall=(1500,3000), preferred_soils=["Loamy","Clay"]),
    "Cardamom":           dict(nitrogen=(60,100), phosphorus=(30,70), potassium=(60,120), ph=(5.0,6.5), moisture=(65,90), temperature=(15,28), humidity=(70,95), rainfall=(1500,3000), preferred_soils=["Loamy","Peaty"]),
    "Pepper":             dict(nitrogen=(60,100), phosphorus=(30,70), potassium=(60,120), ph=(5.5,7.0), moisture=(60,85), temperature=(20,35), humidity=(65,90), rainfall=(1500,2500), preferred_soils=["Loamy","Clay"]),
    "Ginger":             dict(nitrogen=(60,100), phosphorus=(40,80), potassium=(80,150), ph=(5.5,7.5), moisture=(60,85), temperature=(22,32), humidity=(65,90), rainfall=(1500,3000), preferred_soils=["Loamy","Sandy","Peaty"]),
    "Turmeric":           dict(nitrogen=(60,100), phosphorus=(40,80), potassium=(80,150), ph=(5.5,7.0), moisture=(60,85), temperature=(22,32), humidity=(65,90), rainfall=(1500,2500), preferred_soils=["Loamy","Clay","Silt"]),
}

_DEFAULT_PROFILE = dict(
    nitrogen=(40, 120),
    phosphorus=(30, 80),
    potassium=(40, 100),
    ph=(5.5, 7.5),
    moisture=(35, 75),
    temperature=(18, 38),
    humidity=(40, 80),
    rainfall=(500, 1500),
    preferred_soils=SOIL_TYPES,
)

OUTPUT_FILE = "training_dataset.csv"
SAMPLES_PER_CROP = 180   # BALANCED DATASET TARGET
NOISE_PROBABILITY = 0.18
NOISE_FACTOR = 0.08
HARD_CASE_PROBABILITY = 0.10

COLUMNS = [
    "nitrogen", "phosphorus", "potassium", "ph", "moisture", "soil_type",
    "temperature", "humidity", "rainfall", "season", "region", "recommended_crop",
]

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_crop_locations():
    crop_map = defaultdict(list)
    for season, region_map in HIGH_DEMAND_CROPS.items():
        for region, crops in region_map.items():
            for crop in crops:
                crop_map[crop].append((season, region))
    return crop_map


def _profile(crop):
    return CROP_PROFILES.get(crop, _DEFAULT_PROFILE)


def _clamp(val, lo, hi, as_int=True):
    val = max(lo, min(hi, val))
    return round(val) if as_int else round(val, 1)


def _noisy_value(lo, hi, as_int=True):
    val = random.uniform(lo, hi)
    if random.random() < NOISE_PROBABILITY:
        spread = (hi - lo) * NOISE_FACTOR
        val += random.uniform(-spread, spread)
    return _clamp(val, lo, hi, as_int=as_int)


def _pick_soil(preferred_soils):
    if random.random() < 0.75:
        return random.choice(preferred_soils)
    return random.choice(SOIL_TYPES)


def _make_values(profile, season):
    # Base realistic values
    vals = {
        "nitrogen": _noisy_value(*profile["nitrogen"]),
        "phosphorus": _noisy_value(*profile["phosphorus"]),
        "potassium": _noisy_value(*profile["potassium"]),
        "ph": _noisy_value(*profile["ph"], as_int=False),
    }

    # 🌦️ Season-based environmental realism
    if "Monsoon" in season or "Kharif" in season:
        rainfall = random.uniform(1200, 2500)
        humidity = random.uniform(70, 95)
        temperature = random.uniform(22, 32)

    elif "Summer" in season or "Zaid" in season:
        rainfall = random.uniform(200, 800)
        humidity = random.uniform(30, 60)
        temperature = random.uniform(28, 42)

    elif "Winter" in season or "Rabi" in season:
        rainfall = random.uniform(300, 1000)
        humidity = random.uniform(40, 70)
        temperature = random.uniform(10, 25)

    else:
        rainfall = _noisy_value(*profile["rainfall"])
        humidity = _noisy_value(*profile["humidity"])
        temperature = _noisy_value(*profile["temperature"])

    # 🌱 Moisture depends on rainfall
    moisture = int(rainfall * random.uniform(0.03, 0.06))
    moisture = _clamp(moisture, *profile["moisture"])

    vals.update({
        "temperature": round(temperature),
        "humidity": round(humidity),
        "rainfall": round(rainfall),
        "moisture": moisture
    })

    return vals


def generate_row(crop, season, region):
    profile = _profile(crop)

    vals = _make_values(profile, season)

    return {
        "nitrogen": vals["nitrogen"],
        "phosphorus": vals["phosphorus"],
        "potassium": vals["potassium"],
        "ph": vals["ph"],
        "moisture": vals["moisture"],
        "soil_type": _pick_soil(profile["preferred_soils"]),
        "temperature": vals["temperature"],
        "humidity": vals["humidity"],
        "rainfall": vals["rainfall"],
        "season": season,
        "region": region,
        "recommended_crop": crop,
    }
# ──────────────────────────────────────────────────────────────────────────────
# MAIN GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_dataset():
    crop_locations = get_crop_locations()
    rows = []

    print("=" * 70)
    print("GENERATING BALANCED DATASET")
    print("=" * 70)

    for crop, valid_locations in crop_locations.items():
        for _ in range(SAMPLES_PER_CROP):
            season, region = random.choice(valid_locations)
            rows.append(generate_row(crop, season, region))

    random.shuffle(rows)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset saved to      : {OUTPUT_FILE}")
    print(f"Total rows generated  : {len(rows)}")
    print(f"Unique crops          : {len(crop_locations)}")
    print(f"Samples per crop      : {SAMPLES_PER_CROP}")
    print(f"Expected balance      : Strongly balanced")
    print("=" * 70)

    print("\nCrop distribution target:")
    for crop in sorted(crop_locations.keys()):
        print(f" - {crop}: {SAMPLES_PER_CROP}")


if __name__ == "__main__":
    random.seed(42)
    generate_dataset()