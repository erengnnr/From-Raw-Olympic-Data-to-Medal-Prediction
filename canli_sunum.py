import joblib
import time
import random

print("System and Data Loading...\n")

# 1. Load the model and components from .pkl file (No training, opens instantly)
sistem = joblib.load('olimpiyat_sistemi.pkl')
model = sistem['model']

# 2. Load the previously saved test data
X_test_adv, gercek_veriler = joblib.load('sunum_verisi.pkl')

print("="*70)
print("🌟 OLYMPIC DATA SCIENCE: LIVE PREDICTION DEMO 🌟")
print("="*70)

# Select a random athlete
random_idx = random.choice(X_test_adv.index)
sporcu_gercek_veri = gercek_veriler.loc[random_idx]
tek_ornek = X_test_adv.loc[[random_idx]]

print("\nSelected Athlete's Features:")
print(f"Sport: {sporcu_gercek_veri['Sport']}")
print(f"Event: {sporcu_gercek_veri['Event']}")
print(f"Gender: {sporcu_gercek_veri['Sex']}")
print(f"Age: {sporcu_gercek_veri['Age']}")
print(f"Height / Weight: {sporcu_gercek_veri['Height']:.1f} cm / {sporcu_gercek_veri['Weight']:.1f} kg")
print(f"Previous Medals: {sporcu_gercek_veri['Previous_Medals']}")

print("\nModel is analyzing", end="")
for _ in range(4):
    time.sleep(0.5) # Fake waiting time to add realism to the presentation
    print(".", end="", flush=True)

# Make prediction
tahmin = model.predict(tek_ornek)[0]
gercek_sonuc = sporcu_gercek_veri['Medal_Won']

print("\n\n🎯 PREDICTION RESULT:")
if tahmin == 1:
    print("🥇 Model's Prediction: This athlete WILL WIN A MEDAL!")
else:
    print("❌ Model's Prediction: This athlete WILL NOT WIN A MEDAL.")

print("-" * 40)
if gercek_sonuc == 1:
    print("Actual Result: The athlete WON A MEDAL! 🥇")
else:
    print("Actual Result: The athlete DID NOT WIN A MEDAL. ❌")

if tahmin == gercek_sonuc:
    print("✅ Model Predicted Correctly!")
else:
    print("⚠️ Model Was Wrong (There is no 100% accuracy in Data Science!).")
print("="*70)