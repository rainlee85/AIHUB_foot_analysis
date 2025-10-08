#!/usr/bin/env python3
"""
Quick test of geometric analysis framework
"""

from geometric_analysis_framework import GeometricAnalyzer

def main():
    # Test with reduced iterations for speed
    analyzer = GeometricAnalyzer(
        "test_data/output/procrustes_results/aligned_bonewise.feather",
        alignment_type="bonewise",
        n_bootstrap=100,  # Reduced for testing
        n_permutation=100  # Reduced for testing
    )

    data = analyzer.load_data()
    print(f"Loaded {len(data)} landmarks from {data['subject_id'].nunique()} subjects")
    print(f"Disease groups: {data['label_disease'].value_counts().to_dict()}")

    # Quick bone size test
    print("\nComputing bone sizes...")
    sizes_df = analyzer.compute_bone_sizes()
    print(f"Bone sizes computed for {len(sizes_df)} subjects")

    # Show sample bone sizes
    print("\nSample bone sizes:")
    size_cols = [col for col in sizes_df.columns if col.endswith('_size')]
    print(sizes_df[['subject_id', 'label_disease'] + size_cols[:4]].head())

    # Test relative scale ratio analysis on small subset
    print("\nRunning scale ratio analysis (quick test)...")
    ratio_results = analyzer.analyze_relative_scale_ratios()

    print(f"Analysis complete!")

    # Show results summary
    by_disease = ratio_results['comparisons'].get('by_disease', {})
    print(f"Total comparisons: {len(by_disease)}")

    # Show significant results
    significant_ratios = []
    for key, data in by_disease.items():
        stats = data['statistical_results']
        if stats.get('p_value', 1.0) < 0.05:  # Use uncorrected p-value for quick test
            significant_ratios.append({
                'bone_pair': data['bone_pair'],
                'target_group': data['target_group'],
                'ratio_change': data['ratio_change'],
                'effect_size': stats['effect_size'],
                'p_value': stats['p_value']
            })

    if significant_ratios:
        print(f"\nFound {len(significant_ratios)} significant bone ratio changes (p<0.05):")
        for result in significant_ratios[:5]:
            print(f"  {result['bone_pair']} in {result['target_group']}: "
                  f"{result['ratio_change']:.3f}x change "
                  f"(g={result['effect_size']:.3f}, p={result['p_value']:.4f})")
    else:
        print("\nNo significant ratio changes found.")

    # Export quick results
    analyzer.export_results_to_csv(ratio_results, "quick_test_results")
    print("Results exported to quick_test_results/")

if __name__ == "__main__":
    main()