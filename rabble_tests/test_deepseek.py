import os
from dotenv import load_dotenv
from pathlib import Path

# Try to load the .env file
load_dotenv(Path(__file__).parent.parent / "rabble" / ".env")

# Check if the variable is loaded
print(f"DEEPSEEK_API_KEY: {os.getenv('DEEPSEEK_API_KEY')}")
