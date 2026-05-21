from dotenv import load_dotenv
load_dotenv()

import subprocess
subprocess.run(["streamlit", "run", "dashboard.py"])
