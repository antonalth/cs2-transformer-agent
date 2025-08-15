import subprocess
import os

# Step 0: Change working directory to broker/
os.chdir("broker")

# Step 1: Install dependencies
subprocess.run(["npm", "install"], shell=True, check=True)

# Step 2: Run the server with tsx
subprocess.run(["npx", "tsx", r"src\server.ts"], shell=True, check=True)
