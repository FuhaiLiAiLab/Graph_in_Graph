import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'processed_data' / 'scripts'))
from prepare_data import main
if __name__ == "__main__":main()
