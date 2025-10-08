#!/usr/bin/env python3
"""
Complete Geometric Analysis Runner
Performs all three geometric analyses with statistical testing:
1. Translation - Centroid displacement patterns
2. Rotation - Bone orientation changes
3. Aspect Ratios - Height/width elongation patterns
4. Relative Ratios - Inter-bone size relationships
"""

from geometric_analysis_framework import GeometricAnalyzer
import pandas as pd

def run_complete_analysis(data_path, alignment_type, output_dir, n_bootstrap=100, n_permutation=100):
    """Run all available geometric analyses"""

    print(f"\n{'='*70}")
    print(f"COMPLETE GEOMETRIC ANALYSIS - {alignment_type.upper()} ALIGNMENT")
    print(f"{'='*70}\n")

    # Initialize analyzer
    analyzer = GeometricAnalyzer(
        data_path,
        alignment_type=alignment_type,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation
    )

    # Load data
    data = analyzer.load_data()
    print(f"Dataset: {len(data)} landmarks from {data['subject_id'].nunique()} subjects")
    print(f"Available analyses: {analyzer.available_analyses}\n")

    all_results = {}

    # 1. Translation Analysis (global only)
    if 'translation' in analyzer.available_analyses:
        print("1. TRANSLATION ANALYSIS (Centroid Displacement)")
        print("-" * 50)
        translation_results = analyzer.analyze_translation_patterns()
        analyzer.export_results_to_csv(translation_results, f"{output_dir}/translation")

        # Show summary
        by_disease = translation_results['comparisons'].get('by_disease', {})
        sig_count = sum(1 for key, data in by_disease.items()
                       if data.get('displacement_magnitude', {}).get('significant', False))
        print(f"✓ Found {sig_count} significant displacement patterns")
        all_results['translation'] = translation_results
        print()

    # 2. Rotation Analysis (global only)
    if 'rotation' in analyzer.available_analyses:
        print("2. ROTATION ANALYSIS (Bone Orientation)")
        print("-" * 50)
        rotation_results = analyzer.analyze_rotation_patterns()
        analyzer.export_results_to_csv(rotation_results, f"{output_dir}/rotation")

        # Show summary
        by_disease = rotation_results['comparisons'].get('by_disease', {})
        sig_count = sum(1 for key, data in by_disease.items()
                       if data.get('orientation_analysis', {}).get('significant', False))
        print(f"✓ Found {sig_count} significant orientation changes")
        all_results['rotation'] = rotation_results
        print()

    # 3. Aspect Ratio Analysis (both)
    if 'relative_scale_ratios' in analyzer.available_analyses:
        print("3. ASPECT RATIO ANALYSIS (Height/Width)")
        print("-" * 50)
        aspect_results = analyzer.analyze_relative_scale_ratios()
        analyzer.export_results_to_csv(aspect_results, f"{output_dir}/aspect_ratios")

        # Show summary
        by_disease = aspect_results['comparisons'].get('by_disease', {})
        sig_count = sum(1 for key, data in by_disease.items()
                       if data.get('statistical_results', {}).get('significant', False))
        print(f"✓ Found {sig_count} significant aspect ratio changes")
        all_results['aspect_ratios'] = aspect_results
        print()

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE - Results saved to {output_dir}/")
    print(f"{'='*70}\n")

    return all_results

def main():
    """Run analyses on both global and bonewise datasets"""

    # Test with reduced iterations for speed
    N_BOOTSTRAP = 100
    N_PERMUTATION = 100

    print("\n" + "="*70)
    print(" COMPLETE FOOT DISEASE GEOMETRIC ANALYSIS SUITE")
    print("="*70)

    # Analyze global alignment (all analyses available)
    print("\n\n>>> PART 1: GLOBAL ALIGNMENT ANALYSIS <<<")
    global_results = run_complete_analysis(
        data_path="test_data/output/procrustes_results/aligned_global.feather",
        alignment_type="global",
        output_dir="complete_results/global",
        n_bootstrap=N_BOOTSTRAP,
        n_permutation=N_PERMUTATION
    )

    # Analyze bonewise alignment (aspect ratios only)
    print("\n\n>>> PART 2: BONEWISE ALIGNMENT ANALYSIS <<<")
    bonewise_results = run_complete_analysis(
        data_path="test_data/output/procrustes_results/aligned_bonewise.feather",
        alignment_type="bonewise",
        output_dir="complete_results/bonewise",
        n_bootstrap=N_BOOTSTRAP,
        n_permutation=N_PERMUTATION
    )

    # Summary comparison
    print("\n\n" + "="*70)
    print(" COMPARATIVE SUMMARY")
    print("="*70)

    print("\nGLOBAL ALIGNMENT provides:")
    print("  • Translation patterns (inter-bone positioning)")
    print("  • Rotation patterns (bone orientations)")
    print("  • Aspect ratios (bone elongation)")

    print("\nBONEWISE ALIGNMENT provides:")
    print("  • Aspect ratios (pure shape elongation, no positioning bias)")

    print("\n" + "="*70)
    print(" All analyses complete! Check complete_results/ for outputs")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
