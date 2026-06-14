import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("harga_beras_palu.csv")

print("--- 5 Data Pertama ---")
print(data.head())

plt.figure(figsize=(12, 6))  
plt.plot(data["bulan"], data["Harga_Beras"], marker='o', color='b', markersize=4, linewidth=1.5)

lokasi_ticks = range(0, len(data["bulan"]), 6)
label_ticks = [data["bulan"][i] for i in lokasi_ticks]
plt.xticks(lokasi_ticks, label_ticks, rotation=30, ha='right', fontsize=9) 

plt.title("Analisis Prediksi Harga Beras Kota Palu", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Periode (Bulan)", fontsize=11, labelpad=10)
plt.ylabel("Harga (Rp / Kg)", fontsize=11, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.5) 
plt.tight_layout()
plt.show()

X = data[["Inflasi", "Curah_Hujan"]]
y = data["Harga_Beras"]

model = LinearRegression()
model.fit(X, y)

print("\nModel berhasil dilatih")

prediksi = model.predict([[0.50, 120]])

print("Prediksi harga beras:", prediksi[0])
