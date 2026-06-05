import pandas as pd
import matplotlib.pyplot as plt
import os

# Point this to your specific training run folder
# Update 'train7' to whatever folder number your latest training is in
run_folder = 'runs/detect/train7' 
csv_path = os.path.join(run_folder, 'results.csv')

# Load the data
results = pd.read_csv(csv_path)
results.columns = [c.strip() for c in results.columns]

# Plot Training and Validation Loss
plt.figure(figsize=(12, 6))

# Plot Box Loss
plt.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss')
plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss', linestyle='--')

# Plot Class Loss
plt.plot(results['epoch'], results['train/cls_loss'], label='Train Class Loss')
plt.plot(results['epoch'], results['val/cls_loss'], label='Val Class Loss', linestyle='--')

plt.title('YOLOv8 Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_curves.png') # This saves the plot as an image in your GHMODEL folder
plt.show()