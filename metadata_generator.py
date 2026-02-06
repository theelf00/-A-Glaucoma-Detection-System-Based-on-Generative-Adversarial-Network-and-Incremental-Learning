import pandas as pd
import os

files = [f for f in os.listdir('data/raw') if f.endswith(('.jpg', '.png'))]
# Logic: Label is 1 if 'glaucoma' is in the name, else 0
df = pd.DataFrame({
    'filename': files,
    'label': [1 if 'glaucoma' in f.lower() else 0 for f in files]
})
df.to_csv('metadata.csv', index=False)