import os
import argparse
from generate_sample_data import generate_sample_data

def main():
    parser = argparse.ArgumentParser(description="Download or generate SolarCast data")
    parser.add_argument('--output-dir', type=str, default='data/sample', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['download', 'generate'], default='generate', help='Mode to run')
    args = parser.parse_args()

    if args.mode == 'generate':
        print(f"Generating synthetic sample data in {args.output_dir}...")
        generate_sample_data(args.output_dir, num_samples=20)
        print("Done.")
    else:
        print("""
        To download real SDO/HMI SHARP data:
        1. Access JSOC (Joint Science Operations Center) at http://jsoc.stanford.edu/
        2. Query the hmi.sharp_cea_720s series.
        3. For a ready-to-use processed dataset, see Galvez et al. 2019 (SpaceWeather).
        4. Or run this script in 'generate' mode to create synthetic data for testing.
        """)

if __name__ == '__main__':
    main()
