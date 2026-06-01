import os
import json
import shutil

# --- CONFIGURATION (Adjust paths if necessary) ---
AIRFLOW_HOME = os.path.expanduser("~/airflow")
PLANS_DIR = os.path.join(AIRFLOW_HOME, "plans")
WAITING_ROOM = os.path.join(AIRFLOW_HOME, "waiting_room")
STATE_FILE = os.path.join(AIRFLOW_HOME, "cluster_state.json")
MATRIX_DATA = os.path.join(AIRFLOW_HOME, "data/matrix_test")

def reset_environment():
    print("🚀 Starting Cluster Reset...")

    # 1. Reset Cluster State
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'w') as f:
            json.dump({
                "PT": {},
                "ES": {},
                "DE": {},
                "PL": {},
                "SW": {}
            }, f)
        print(f"✅ Reset {STATE_FILE} to empty state.")
    else:
        print(f"⚠️ State file not found at {STATE_FILE}")

    # 2. Clear Plans Folder (JSONs and Heatmaps)
    if os.path.exists(PLANS_DIR):
        files_removed = 0
        for filename in os.listdir(PLANS_DIR):
            file_path = os.path.join(PLANS_DIR, filename)
            try:
                if (os.path.isfile(file_path) or os.path.islink(file_path)) and not file_path.endswith('.png'):
                    os.unlink(file_path)
                    files_removed += 1
            except Exception as e:
                print(f'❌ Failed to delete {file_path}. Reason: {e}')
        print(f"✅ Removed {files_removed} files from {PLANS_DIR}")

    # 3. Clear Waiting Room
    if os.path.exists(WAITING_ROOM):
        for filename in os.listdir(WAITING_ROOM):
            file_path = os.path.join(WAITING_ROOM, filename)
            os.unlink(file_path)
        print(f"✅ Waiting room cleared.")

    # 4. Clear Matrix Data (Optional: prevents disk filling up)
    if os.path.exists(MATRIX_DATA):
        shutil.rmtree(MATRIX_DATA)
        os.makedirs(MATRIX_DATA)
        print(f"✅ Matrix .npy data cleared.")

    print("\n✨ Environment is clean. You can now run a new experiment!")

if __name__ == "__main__":
    reset_environment()