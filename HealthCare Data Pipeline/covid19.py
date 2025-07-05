import numpy as np
# Load the dataset
data = np.genfromtxt('covid_19_india.csv', delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8), dtype=None, encoding=None, names=True)

# Find missing values
valid_mask = (data['StateUnionTerritory'] != 'Unassigned') & (data['StateUnionTerritory'] != 'Cases being reassigned to states')
data = data[valid_mask] 

# Clean numerical data
cured = np.nan_to_num(data['Cured'], nan = 0.0)
deaths = np.nan_to_num(data['Deaths'], nan = 0.0)
confirmed = np.nan_to_num(data['Confirmed'], nan = 0.0)

# Handle negative daily_new_cases
daily_new_cases = np.abs(np.diff(confirmed))
daily_deaths = np.abs(np.diff(deaths))

# Time-Based: Peak Death Date
peak_deaths_idx = np.argmax(deaths)
peak_death_date = data['Date'][peak_deaths_idx]
peak_deaths = deaths[peak_deaths_idx]

# Statistical Summary
national_death_rate = np.sum(deaths)/np.sum(confirmed) * 100
national_recovery_rate = np.sum(cured)/np.sum(confirmed) * 100

# State with Worst Recovery
state_names = np.unique(data['StateUnionTerritory'])
recovery_rate = [np.sum(cured[data['StateUnionTerritory'] == s]) / np.sum(confirmed[data['StateUnionTerritory'] == s]) for s in state_names]
worst_recovery_idx = np.nanargmin(recovery_rate)
worst_recovery_state = state_names[worst_recovery_idx]
worst_recovery_rate = recovery_rate[worst_recovery_idx] * 100

print("\nNATIONAL SUMMARY")
print(f"Peak Deaths: {peak_deaths} cases on {peak_death_date}")
print(f"National Death Rate: {national_death_rate:.2f}%")
print(f"National Recovery Rate: {national_recovery_rate:.2f}%")
print(f"State with Lowest Recovery: {worst_recovery_state} with recovery rate of ({worst_recovery_rate:.2f}%)\n")
if worst_recovery_rate == 0:
    print("Note: 0.00% recovery may indicate missing data for this region\n ")

# State wise analysis
high_risk_states = set()
highest_ratio = 0
worst_ratio_state = None
for state in np.unique(data['StateUnionTerritory']) :
    # Masks
    state_mask = data['StateUnionTerritory'] == state
    state_cured = cured[state_mask]
    state_confirmed = confirmed[state_mask]
    state_deaths = deaths[state_mask]

    # Peak analysis
    peak_cases = np.max(state_confirmed)
    peak_deaths = np.max(state_deaths)

    # Death Rate
    with np.errstate(divide='ignore', invalid='ignore'):
        death_rate = np.divide(state_deaths, state_confirmed) * 100
        death_rate = np.nan_to_num(death_rate, nan = 0.0)
        death_to_cured_ratio = np.divide(state_deaths, state_cured)
        death_to_cured_ratio = np.nan_to_num(death_to_cured_ratio , nan = 0.0, posinf = 0.0)
    
    # Risk Label
    death_risk = "High" if np.mean(death_rate) > 2 else "Low"
    if death_risk == 'High' :
        high_risk_states.add(state)
    
    # Death to Cured Ratio
    valid_ratios = death_to_cured_ratio[np.isfinite(death_to_cured_ratio) & (death_to_cured_ratio > 0)]
    if len(valid_ratios) > 0 :
        current_ratio = np.mean(valid_ratios)
        if current_ratio > highest_ratio:
            highest_ratio = current_ratio
            worst_ratio_state = state

    print(f"{state} :")
    print(f"Peak cases = {peak_cases}")
    print(f"Peak deaths = {peak_deaths}")
    print(f"Death rate = {np.mean(death_rate):.2f}% (Risk: {death_risk})")
    print(f"Death-to-Cured ratio = {np.mean(valid_ratios) if len(valid_ratios) > 0 else 'N/A'}\n")
    
print("Final Results : ")
high_risk_states = [str(state) for state in high_risk_states]
print("High-risk states:", sorted(high_risk_states))
print(f"Worst State: {worst_ratio_state} (Death-to-Cured Ratio: {highest_ratio:.2f})")