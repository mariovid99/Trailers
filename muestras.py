import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# === FECHAS ===
start_date = datetime(2024, 1, 1)
today = datetime(2025, 6, 12)
end_date = datetime(2025, 12, 31)
n_days_past = (today - start_date).days + 1
n_days_future = (end_date - today).days
dates_past = [start_date + timedelta(days=i) for i in range(n_days_past)]
dates_future = [today + timedelta(days=i + 1) for i in range(n_days_future)]
dates_full = dates_past + dates_future

# === PARÁMETROS ===
threshold = 130
real_maint_past = np.linspace(100, n_days_past - 50, 7, dtype=int)
real_maint_future = np.linspace(30, n_days_future - 20, 3, dtype=int)

# === SIMULACIÓN ===
np.random.seed(42)
sensor_names = ['Temperatura', 'Vibracion', 'Presion']
sensor_data, ai_preds, classic_preds = {}, {}, {}

for name in sensor_names:
    past_signal = (
        np.linspace(50, 100, n_days_past) +
        np.random.normal(0, 8, n_days_past) +
        np.sin(np.linspace(0, 10 * np.pi, n_days_past)) * 10
    )
    past_signal[real_maint_past] = threshold + 10

    future_signal = (
        past_signal[-1] +
        np.linspace(0, 15, n_days_future) +
        np.sin(np.linspace(0, 6 * np.pi, n_days_future)) * 8 +
        np.random.normal(0, 5, n_days_future)
    )
    future_signal[real_maint_future] = threshold + 12

    signal = np.concatenate([past_signal, future_signal])
    sensor_data[name] = signal

    ai_past = past_signal + np.random.normal(0, 3, n_days_past) - 2
    ai_past[real_maint_past[0]] = past_signal[real_maint_past[0]] - 25
    ai_future = future_signal + np.random.normal(0, 2, n_days_future) - 1
    ai_preds[name] = np.concatenate([ai_past, ai_future])

    classic_past = past_signal + np.random.normal(0, 12, n_days_past) - 10
    classic_extra_past = np.sort(np.random.choice(np.setdiff1d(np.arange(n_days_past), real_maint_past), 10, replace=False))
    classic_past[classic_extra_past] = threshold + 20

    classic_future = future_signal + np.random.normal(0, 8, n_days_future) - 5
    classic_extra_future = np.sort(np.random.choice(np.setdiff1d(np.arange(n_days_future), real_maint_future), 5, replace=False))
    classic_future[classic_extra_future] = threshold + 20

    classic_preds[name] = np.concatenate([classic_past, classic_future])



# === GRAFICO 1: REAL VS CLASICO (3 PANELES) ===
fig1, axs1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i, name in enumerate(sensor_names):
    axs1[i].plot(dates_past, sensor_data[name][:n_days_past], color='black', linewidth=2, label='Real')
    axs1[i].plot(dates_future, sensor_data[name][n_days_past:], color='black', linewidth=2, linestyle='--')
    axs1[i].plot(dates_past, classic_preds[name][:n_days_past], color='royalblue', linewidth=2, label='Clásico')
    axs1[i].plot(dates_future, classic_preds[name][n_days_past:], color='royalblue', linewidth=2, linestyle='--')
    axs1[i].axvline(today, color='yellow', linestyle='--', linewidth=2, label='Hoy')
    axs1[i].set_title(f'{name}: Real vs Clásico')
    axs1[i].grid(True)
    if i == 0:
        axs1[i].legend()
plt.xlabel("Fecha")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle("Comparación Real vs Clásico por Sensor", fontsize=16)
plt.show()

# === GRAFICO 2: REAL VS IA (3 PANELES) ===
fig2, axs2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i, name in enumerate(sensor_names):
    axs2[i].plot(dates_past, sensor_data[name][:n_days_past], color='black', linewidth=2, label='Real')
    axs2[i].plot(dates_future, sensor_data[name][n_days_past:], color='black', linewidth=2, linestyle='--')
    axs2[i].plot(dates_past, ai_preds[name][:n_days_past], color='darkorange', linewidth=2, label='IA')
    axs2[i].plot(dates_future, ai_preds[name][n_days_past:], color='darkorange', linewidth=2, linestyle='--')
    axs2[i].axvline(today, color='yellow', linestyle='--', linewidth=2, label='Hoy')
    axs2[i].set_title(f'{name}: Real vs IA')
    axs2[i].grid(True)
    if i == 0:
        axs2[i].legend()
plt.xlabel("Fecha")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle("Comparación Real vs IA por Sensor", fontsize=16)
plt.show()


# === GRÁFICO 3: FALLAS - IA VS REAL ===
fig3, ax3 = plt.subplots(figsize=(14, 5))
name = 'Temperatura'
true_data = sensor_data[name]
ai_pred = ai_preds[name]

real_flags_past = true_data[:n_days_past] > threshold
real_flags_future = true_data[n_days_past:] > threshold
ai_flags_future = ai_pred[n_days_past:] > threshold

ax3.plot(dates_past, true_data[:n_days_past], label='Real', color='black', linewidth=1.5)
ax3.plot(dates_future, true_data[n_days_past:], color='black', linewidth=1.5, linestyle='--')

ax3.plot(dates_past, ai_pred[:n_days_past], label='IA', color='darkorange', linewidth=2)
ax3.plot(dates_future, ai_pred[n_days_past:], color='darkorange', linewidth=2, linestyle='--')

ax3.axhline(threshold, color='red', linestyle='--', label='Umbral mantenimiento')
ax3.axvline(today, color='yellow', linestyle='--', linewidth=2, label='Hoy')

ax3.fill_between(dates_future, threshold, ai_pred[n_days_past:], where=ai_flags_future, color='darkorange', alpha=0.2)
ax3.scatter(np.array(dates_past)[real_flags_past], true_data[:n_days_past][real_flags_past], color='black', marker='o', label='Falla real (hist)')
ax3.scatter(np.array(dates_future)[real_flags_future], true_data[n_days_past:][real_flags_future], color='black', marker='x', label='Falla real (fut)')

ax3.set_title('Predicción de Fallas: IA vs Real (Temperatura)')
ax3.legend()
ax3.grid(True)
plt.xlabel("Fecha")
plt.tight_layout()
plt.show()

# === GRÁFICO 4: FALLAS - CLÁSICO VS REAL ===
fig4, ax4 = plt.subplots(figsize=(14, 5))
classic_pred = classic_preds[name]
classic_flags_future = classic_pred[n_days_past:] > threshold

ax4.plot(dates_past, true_data[:n_days_past], label='Real', color='black', linewidth=1.5)
ax4.plot(dates_future, true_data[n_days_past:], color='black', linewidth=1.5, linestyle='--')

ax4.plot(dates_past, classic_pred[:n_days_past], label='Clásico', color='royalblue', linewidth=2)
ax4.plot(dates_future, classic_pred[n_days_past:], color='royalblue', linewidth=2, linestyle='--')

ax4.axhline(threshold, color='red', linestyle='--', label='Umbral mantenimiento')
ax4.axvline(today, color='yellow', linestyle='--', linewidth=2, label='Hoy')

ax4.fill_between(dates_future, threshold, classic_pred[n_days_past:], where=classic_flags_future, color='royalblue', alpha=0.2)
ax4.scatter(np.array(dates_past)[real_flags_past], true_data[:n_days_past][real_flags_past], color='black', marker='o', label='Falla real (hist)')
ax4.scatter(np.array(dates_future)[real_flags_future], true_data[n_days_past:][real_flags_future], color='black', marker='x', label='Falla real (fut)')

ax4.set_title('Predicción de Fallas: Clásico vs Real (Temperatura)')
ax4.legend()
ax4.grid(True)
plt.xlabel("Fecha")
plt.tight_layout()
plt.show()


#Export data
import json

# Antes de exportar, calcula los flags de fallas
json_data = {
    "dates": [d.strftime("%Y-%m-%d") for d in dates_full],
    "real": {name: sensor_data[name].tolist() for name in sensor_names},
    "ia": {name: ai_preds[name].tolist() for name in sensor_names},
    "clasico": {name: classic_preds[name].tolist() for name in sensor_names},
    "threshold": threshold,
    # Nueva información para gráficos 3 y 4:
    "real_failures": {
        "past": [dates_past[i].strftime("%Y-%m-%d") for i in np.where(sensor_data['Temperatura'][:n_days_past] > threshold)[0]],
        "future": [dates_future[i].strftime("%Y-%m-%d") for i in np.where(sensor_data['Temperatura'][n_days_past:] > threshold)[0]]
    },
    "ia_failures": {
        "future": [dates_future[i].strftime("%Y-%m-%d") for i in np.where(ai_preds['Temperatura'][n_days_past:] > threshold)[0]]
    },
    "classic_failures": {
        "future": [dates_future[i].strftime("%Y-%m-%d") for i in np.where(classic_preds['Temperatura'][n_days_past:] > threshold)[0]]
    }
}

with open("dashboard_data.json", "w") as f:
    json.dump(json_data, f)

# === TABLAS DE RESUMEN ===

summary_past = {
    'Mantenimientos Reales (hasta hoy)': np.sum(real_flags_past),
    'Detectados por IA': np.sum(ai_pred[:n_days_past] > threshold),
    'Detectados por Clásica': np.sum(classic_pred[:n_days_past] > threshold)
}
summary_past['Ahorro con IA'] = summary_past['Detectados por Clásica'] - summary_past['Detectados por IA']
summary_past['Precisión IA'] = round(summary_past['Mantenimientos Reales (hasta hoy)'] / summary_past['Detectados por IA'], 2)

summary_future = {
    'Mantenimientos Reales (resto 2025)': np.sum(real_flags_future),
    'Proyectados por IA': np.sum(ai_flags_future),
    'Proyectados por Clásica': np.sum(classic_flags_future),
    'Diferencia de Proyecciones': np.sum(classic_flags_future) - np.sum(ai_flags_future)
}
summary_future['Precisión IA (futuro)'] = round(np.sum(real_flags_future) / summary_future['Proyectados por IA'], 2)

tabla_past = pd.DataFrame([summary_past])
tabla_future = pd.DataFrame([summary_future])

print("=== Resumen Histórico ===")
print(tabla_past.to_string(index=False))

print("\n=== Proyecciones Futuras ===")
print(tabla_future.to_string(index=False))


