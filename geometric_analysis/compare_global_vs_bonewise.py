#!/usr/bin/env python3
"""
Comprehensive comparison of global vs bonewise alignment analysis
"""

from geometric_analysis_framework import GeometricAnalyzer
import pandas as pd

def analyze_dataset(data_path, alignment_type, output_prefix):
    """Analyze a single dataset and return results summary"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {alignment_type.upper()} ALIGNMENT")
    print(f"{'='*60}")

    # Initialize analyzer
    analyzer = GeometricAnalyzer(
        data_path,
        alignment_type=alignment_type,
        n_bootstrap=100,  # Quick test
        n_permutation=100
    )

    # Load data
    data = analyzer.load_data()
    print(f"Dataset: {len(data)} landmarks from {data['subject_id'].nunique()} subjects")
    print(f"Available analyses: {analyzer.available_analyses}")

    disease_counts = data['label_disease'].value_counts()
    print(f"Disease distribution: {disease_counts.to_dict()}")

    results_summary = {
        'alignment_type': alignment_type,
        'available_analyses': analyzer.available_analyses,
        'disease_groups': list(disease_counts.index),
        'sample_sizes': disease_counts.to_dict()
    }

    # Run analyses available for this alignment type
    if 'translation' in analyzer.available_analyses:
        print("\n--- TRANSLATION ANALYSIS ---")
        try:
            translation_results = analyzer.analyze_translation_patterns()
            analyzer.export_results_to_csv(translation_results, f"{output_prefix}_translation")

            # Count significant results
            by_disease = translation_results['comparisons'].get('by_disease', {})
            significant_count = 0
            for key, data in by_disease.items():
                for coord in ['x_coordinate', 'y_coordinate', 'displacement_magnitude']:
                    if coord in data and data[coord].get('significant', False):
                        significant_count += 1

            results_summary['translation_significant'] = significant_count
            print(f"Translation analysis: {significant_count} significant results")

        except Exception as e:
            print(f"Translation analysis failed: {e}")
            results_summary['translation_significant'] = 0

    if 'relative_scale_ratios' in analyzer.available_analyses:
        print("\n--- RELATIVE SCALE RATIO ANALYSIS ---")
        try:
            ratio_results = analyzer.analyze_relative_scale_ratios()
            analyzer.export_results_to_csv(ratio_results, f"{output_prefix}_ratios")

            # Count significant results
            by_disease = ratio_results['comparisons'].get('by_disease', {})
            significant_ratios = []
            for key, data in by_disease.items():
                stats = data['statistical_results']
                if stats.get('significant', False):
                    significant_ratios.append({
                        'bone_pair': data['bone_pair'],
                        'target_group': data['target_group'],
                        'ratio_change': data['ratio_change'],
                        'effect_size': stats['effect_size'],
                        'p_value': stats['p_value']
                    })

            results_summary['ratio_significant'] = len(significant_ratios)
            results_summary['significant_ratios'] = significant_ratios[:5]  # Top 5

            print(f"Ratio analysis: {len(significant_ratios)} significant results")
            if significant_ratios:
                print("Top significant bone ratio changes:")
                for result in significant_ratios[:3]:
                    print(f"  {result['bone_pair']} in {result['target_group']}: "
                          f"{result['ratio_change']:.3f}x (g={result['effect_size']:.3f}, p={result['p_value']:.3f})")

        except Exception as e:
            print(f"Ratio analysis failed: {e}")
            results_summary['ratio_significant'] = 0

    return results_summary

def main():
    print("GEOMETRIC ANALYSIS COMPARISON: GLOBAL vs BONEWISE")
    print("Comparing geometric variation patterns between alignment strategies")

    # Analyze both datasets
    global_results = analyze_dataset(
        "test_data/output/procrustes_results/aligned_global.feather",
        "global",
        "global_results"
    )

    bonewise_results = analyze_dataset(
        "test_data/output/procrustes_results/aligned_bonewise.feather",
        "bonewise",
        "bonewise_results"
    )

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    print(f"Global alignment available analyses: {global_results['available_analyses']}")
    print(f"Bonewise alignment available analyses: {bonewise_results['available_analyses']}")

    print(f"\nSIGNIFICANT FINDINGS:")
    if 'translation_significant' in global_results:
        print(f"Global - Translation patterns: {global_results['translation_significant']} significant")
    if 'ratio_significant' in global_results:
        print(f"Global - Scale ratios: {global_results['ratio_significant']} significant")
    if 'ratio_significant' in bonewise_results:
        print(f"Bonewise - Scale ratios: {bonewise_results['ratio_significant']} significant")

    print(f"\nCLINICAL INTERPRETATION:")
    print("- Global alignment: Reveals inter-bone positioning and orientation changes")
    print("- Bonewise alignment: Shows pure shape changes and relative bone size patterns")
    print("- Both approaches provide complementary insights into disease mechanisms")

    # Show bone ratio differences between datasets
    if bonewise_results.get('significant_ratios'):
        print(f"\nBONEWISE SIGNIFICANT RATIOS:")
        for result in bonewise_results['significant_ratios']:
            print(f"  {result['bone_pair']}: {result['ratio_change']:.4f}x change in {result['target_group']}")

if __name__ == "__main__":
    main()