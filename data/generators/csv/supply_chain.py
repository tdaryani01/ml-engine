import os
import numpy as np
import pandas as pd

def generate_industrial_supply_dataset(n_samples=3000):
    np.random.seed(101)
    
    # 1. Core independent continuous industrial drivers
    lead_time_days = np.random.uniform(5, 60, n_samples)
    logistics_delay_idx = np.random.exponential(scale=1.5, size=n_samples)
    resource_reserve_pct = np.random.beta(a=5, b=2, size=n_samples) * 100
    labor_availability = np.random.uniform(0.6, 1.0, n_samples)
    macro_inflation_rate = np.random.normal(loc=3.2, scale=1.1, size=n_samples)
    
    # 2. Latent coupled systemic stress score calculation
    # Simulates cascading non-linear risk accumulation
    stress_score = (
        (lead_time_days / 15.0) ** 1.8 +
        (logistics_delay_idx * 2.5) ** 1.3 -
        (resource_reserve_pct / 20.0) +
        (1.0 - labor_availability) * 8.0 +
        (macro_inflation_rate * 0.4)
    )
    
    # Add stochastic operational noise
    stress_score += np.random.normal(loc=0, scale=1.2, size=n_samples)
    
    # 3. Multi-Class Operational Health Mapping (Target)
    # 0: Stable (Green), 1: Stressed (Yellow), 2: Critical Disruption (Red)
    target = np.zeros(n_samples, dtype=int)
    target[stress_score > 6.5] = 1
    target[stress_score > 12.0] = 2
    
    df = pd.DataFrame({
        'Supplier_Lead_Time': lead_time_days,
        'Logistics_Delay_Index': logistics_delay_idx,
        'Resource_Reserve_Percent': resource_reserve_pct,
        'Labor_Capacity_Utilization': labor_availability,
        'Macro_Inflation_Rate': macro_inflation_rate,
        'Target': target
    })
    
    os.makedirs("./data/generator", exist_ok=True)
    df.to_csv("./data/generator/supply_chain.csv", index=False)
    print(f"Industrial Datastream generated successfully at ./data/generator/supply_chain.csv")
    print(f"Class Distribution - Stable: {sum(target==0)}, Stressed: {sum(target==1)}, Critical: {sum(target==2)}")

if __name__ == "__main__":
    generate_industrial_supply_dataset()