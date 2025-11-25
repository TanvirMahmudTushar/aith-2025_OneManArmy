

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main inference module
from Inference.infer import main

if __name__ == "__main__":
    # Execute inference (banner is printed in main)
    main()

