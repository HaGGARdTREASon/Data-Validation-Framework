import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
pred_df = pd.read_csv('pred_data_for_excel.csv')
sim_df = pd.read_csv('sim_data_for_excel.csv')

# 2. Setup the main figure
fig, ax_main = plt.subplots(figsize=(8, 6))

# Plot Main Data
ax_main.plot(sim_df['Length'], sim_df['Sim_Sens_Scaled'], 
             label='Simulation Data', color='#1f77b4', linewidth=2)
ax_main.scatter(pred_df['Red_X'], pred_df['Red_Y_Scaled'], 
                label='Predicted Data', color='#d62728', s=15, alpha=0.6)

# Formatting Main Plot
ax_main.set_xlabel(r'Length / Wavelength ($nm$)', fontsize=12)
ax_main.set_ylabel(r'Scaled Sensitivity (a.u.)', fontsize=12)
ax_main.set_title('Environmental Sensor Data Comparison', fontsize=14)
ax_main.legend(loc='upper right')
ax_main.grid(True, linestyle='--', alpha=0.6)

# 3. Create the Inset Plot
# Coordinates for inset: [left, bottom, width, height] relative to figure
ax_inset = fig.add_axes([0.55, 0.45, 0.3, 0.3]) 

ax_inset.plot(sim_df['Length'], sim_df['Sim_Sens_Scaled'], color='#1f77b4', linewidth=1.5)
ax_inset.scatter(pred_df['Red_X'], pred_df['Red_Y_Scaled'], color='#d62728', s=10, alpha=0.5)

# Focus the inset on a specific range
ax_inset.set_xlim(250, 255)
ax_inset.set_ylim(0, 0.4)
ax_inset.set_title('Zoomed View', fontsize=10)
ax_inset.grid(True, linestyle=':', alpha=0.5)
ax_inset.tick_params(labelsize=8)

# Save the high-resolution figure
plt.savefig('environmental_data_plot.png', dpi=300, bbox_inches='tight')
plt.show()