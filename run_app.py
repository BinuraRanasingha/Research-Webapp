# run_streamlit.py
import subprocess

# Specify the name of your Streamlit app script
streamlit_app_script = "streamlit_application.py"

# Build the command to run the Streamlit app
command = ["streamlit", "run", streamlit_app_script]

# Run the Streamlit app using subprocess
subprocess.run(command)
