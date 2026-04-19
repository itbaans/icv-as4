import sys
import os

# Ensure the src directory is in the system path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from evaluate_performance import main as run_performance
from evaluate_ablations import test_internal_ablations as run_ablations

def main():
    print("="*80)
    print("GRAND ORCHESTRATOR: RUNNING FULL EVALUATION PIPELINE")
    print("="*80)
    
    # 1. Run Performance Script
    # Mocking sys.argv so argparse in evaluate_performance doesn't crash expecting external terminal inputs
    sys.argv = ['evaluate_performance.py']
    print("\n\n>>> STARTING PERFORMANCE BENCHMARKS (SECTION 5.2) <<<")
    try:
        run_performance()
    except Exception as e:
        print(f"Performance eval failed: {e}")
        
    # 2. Run Ablation Script
    print("\n\n>>> STARTING FEATURE ABLATIONS (SECTION 5.1) <<<")
    try:
        run_ablations()
    except Exception as e:
        print(f"Ablation eval failed: {e}")
        
    print("\n\n" + "="*80)
    print("FULL PIPELINE EXECUTION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
