import sys
import os

# Add the root directory of your package to the python search path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

#Import and run the Streamlit app from sats4u/sats4ulive.py
from sats4u.sats4ulive import main 

if __name__ == '__main__':
    main()