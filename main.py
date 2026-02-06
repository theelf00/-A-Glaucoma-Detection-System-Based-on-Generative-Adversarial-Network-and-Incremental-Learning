import os
import subprocess

def run_step(script_name):
    print(f"\n>>> Running {script_name}...")
    result = subprocess.run(['python', script_name], capture_output=False)
    if result.returncode != 0:
        print(f"Error occurred in {script_name}. Stopping.")
        exit(1)

def main():
    # 1. Create the CSV for your initial raw images
    # (Assuming you've put ACRIMA/RIM-ONE images in data/raw)
    print("Step 1: Preparing initial metadata...")
    # You can call your csv script logic here directly or as a separate file
    
    # 2. Train the GAN
    run_step('train_gan.py')
    
    # 3. Augment/Balance the data
    # (This adds the synthetic images to data/raw)
    run_step('utils/augment_data.py')
    
    # 4. Update the CSV to include the NEW synthetic images
    print("Step 4: Updating metadata with synthetic images...")
    # (Call the CSV script again)
    
    # 5. Incremental Learning
    run_step('train_incremental.py')
    
    # 6. Final Evaluation
    run_step('evaluate.py')

    print("\n[SUCCESS] Entire Glaucoma Detection Pipeline Completed!")

if __name__ == "__main__":
    main()