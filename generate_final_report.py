"""
Generate Final Survival Analysis Report
"""

import sys
import os
import pandas as pd

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis import compute_survival_summary, format_months, describe_c_index


def _significant_factors(hazard_ratios: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    return hazard_ratios[hazard_ratios['p'] < alpha].sort_values('hr', ascending=False)


def generate_comprehensive_report():
    """Generate comprehensive survival analysis report."""

    # Load the synthetic data
    df = pd.read_csv("data/synthetic_survival_sample.csv")
    results = compute_survival_summary(df)
    site_counts = results['site_counts']
    stage_counts = results['stage_counts']
    significant = _significant_factors(results['hazard_ratios'])
    c_index = results['c_index_test']

    print("=== COMPREHENSIVE SURVIVAL ANALYSIS REPORT ===\n")

    # Basic Statistics
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total Patients: {results['n_patients']:,}")
    print(f"Deaths (Events): {results['n_events']:,} ({results['event_rate']:.1%})")
    print(f"Median Follow-up: {results['median_follow_up']:.0f} months (reverse Kaplan-Meier)")
    print(f"Age Range: {results['age_min']} - {results['age_max']} years")
    print(f"Mean Age: {results['age_mean']:.1f} years")

    # Cancer Sites
    print(f"\nCANCER SITES DISTRIBUTION")
    print("=" * 50)
    for site, count in site_counts.items():
        print(f"{site}: {count:,} patients ({count/len(df):.1%})")

    # Stage Distribution
    print(f"\nSTAGE DISTRIBUTION")
    print("=" * 50)
    for stage, count in stage_counts.items():
        print(f"Stage {stage}: {count:,} patients ({count/len(df):.1%})")

    # Survival by Stage
    print(f"\nSURVIVAL BY STAGE")
    print("=" * 50)
    for stage, data in results['by_stage'].items():
        print(f"Stage {stage}: {data['patients']} patients, median survival: "
              f"{format_months(data['median_survival'])}, 5-year survival: {data['survival_5yr']:.1%}")

    # Survival by Site
    print(f"\nSURVIVAL BY PRIMARY SITE")
    print("=" * 50)
    for site, data in results['by_site'].items():
        print(f"{site}: {data['patients']} patients, median survival: "
              f"{format_months(data['median_survival'])}, 5-year survival: {data['survival_5yr']:.1%}")

    # Age Group Analysis
    print(f"\nSURVIVAL BY AGE GROUP")
    print("=" * 50)
    for age_group, data in results['by_age_group'].items():
        print(f"Age {age_group}: {data['patients']} patients, median survival: "
              f"{format_months(data['median_survival'])}, 5-year survival: {data['survival_5yr']:.1%}")

    # Cox Model Results
    print(f"\nCOX PROPORTIONAL HAZARDS MODEL")
    print("=" * 50)
    print(f"Concordance Index (C-index, 30% test split): {c_index:.3f} "
          f"({describe_c_index(c_index).capitalize()} predictive performance)")
    print("\nSignificant Risk Factors (p < 0.05):")
    for name, row in significant.iterrows():
        print(f"• {name}: HR = {row['hr']:.2f} (95% CI {row['hr_lower']:.2f}-{row['hr_upper']:.2f})")

    # Key Findings
    by_stage = results['by_stage']
    by_site = results['by_site']
    best_site = max(by_site, key=lambda s: by_site[s]['survival_5yr'])
    worst_site = min(by_site, key=lambda s: by_site[s]['survival_5yr'])
    print(f"\nKEY FINDINGS")
    print("=" * 50)
    print(f"1. **Stage is the strongest predictor**: 5-year survival falls from "
          f"{by_stage['I']['survival_5yr']:.0%} (stage I) to {by_stage['IV']['survival_5yr']:.0%} (stage IV)")
    print(f"2. **Primary site matters**: 5-year survival ranges from {by_site[worst_site]['survival_5yr']:.0%} "
          f"({worst_site}) to {by_site[best_site]['survival_5yr']:.0%} ({best_site})")
    print("3. **Age effect**: Older patients have worse outcomes")
    print(f"4. **Overall survival**: {1 - results['event_rate']:.1%} of patients alive at last follow-up")
    print(f"5. **Model performance**: C-index of {c_index:.3f} indicates {describe_c_index(c_index)} discrimination")

    # Data Quality Assessment
    print(f"\nDATA QUALITY ASSESSMENT")
    print("=" * 50)
    print(f"• Missing values: tumor size unknown for {df['tumor_size_mm'].isna().mean():.0%} of patients "
          "(imputed with the median in the Cox model)")
    print("• Site-specific case mix: sex, age, stage and treatment follow published cancer statistics")
    print("• Follow-up: diagnoses 2010-2020, administratively censored at December 2021")
    print(f"• Sample size: {results['n_patients']:,} patients")

    # Clinical Implications
    print(f"\nCLINICAL IMPLICATIONS")
    print("=" * 50)
    print("• Early detection crucial: Localized disease has excellent prognosis")
    print("• Risk stratification: Model can identify high-risk patients")
    print("• Treatment planning: Site, stage and grade should guide therapy decisions")
    print("• Follow-up strategies: More intensive monitoring for advanced stages")

    # Create a summary statistics table
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Patients',
            'Deaths (Events)',
            'Event Rate',
            'Median Follow-up (months)',
            'Mean Age (years)',
            'Stage I Patients',
            'Stage II Patients',
            'Stage III Patients',
            'Stage IV Patients',
            '5-year Overall Survival',
            'C-index (test set)'
        ],
        'Value': [
            f"{results['n_patients']:,}",
            f"{results['n_events']:,}",
            f"{results['event_rate']:.1%}",
            f"{results['median_follow_up']:.0f}",
            f"{results['age_mean']:.1f}",
            f"{stage_counts.get('I', 0)} ({stage_counts.get('I', 0)/len(df):.1%})",
            f"{stage_counts.get('II', 0)} ({stage_counts.get('II', 0)/len(df):.1%})",
            f"{stage_counts.get('III', 0)} ({stage_counts.get('III', 0)/len(df):.1%})",
            f"{stage_counts.get('IV', 0)} ({stage_counts.get('IV', 0)/len(df):.1%})",
            f"{results['overall']['survival_5yr']:.1%}",
            f"{c_index:.3f}"
        ]
    })

    # Save detailed report
    os.makedirs("reports", exist_ok=True)

    def survival_lines(groups, prefix=''):
        return chr(10).join(
            f"  - {prefix}{name}: median {format_months(d['median_survival'])}, "
            f"5-year survival {d['survival_5yr']:.1%} (n={d['patients']})"
            for name, d in groups.items())

    def format_p(p):
        return 'p < 0.001' if p < 0.001 else f"p = {p:.3f}"

    hr_lines = chr(10).join(
        f"  - {name} (HR: {row['hr']:.2f}, 95% CI {row['hr_lower']:.2f}-{row['hr_upper']:.2f}, "
        f"{format_p(row['p'])})"
        for name, row in significant.iterrows())

    report_content = f"""# Comprehensive Survival Analysis Report

## Executive Summary
This analysis used a synthetic cancer survival cohort (n={results['n_patients']:,}) to demonstrate survival analysis methodologies and validate model performance. The cohort's case mix and stage-specific survival are calibrated to published US cancer statistics for six primary sites.

## Dataset Overview
- **Total Patients**: {results['n_patients']:,}
- **Events (Deaths)**: {results['n_events']:,} ({results['event_rate']:.1%})
- **Median Follow-up**: {results['median_follow_up']:.0f} months (reverse Kaplan-Meier)
- **Age Range**: {results['age_min']} - {results['age_max']} years (median {results['age_median']:.0f})
- **Diagnosis Years**: {df['year_of_diagnosis'].min()} - {df['year_of_diagnosis'].max()}, follow-up through December 2021

## Cancer Distribution
### Primary Sites
{chr(10).join([f"- **{site}**: {count:,} patients ({count/len(df):.1%})" for site, count in site_counts.items()])}

### Stage Distribution
{chr(10).join([f"- **Stage {stage}**: {count:,} patients ({count/len(df):.1%})" for stage, count in stage_counts.items()])}

## Survival Analysis Results

### Kaplan-Meier Analysis
- **Overall**: median {format_months(results['overall']['median_survival'])}, 5-year survival {results['overall']['survival_5yr']:.1%}
- **Stage-specific survival**:
{survival_lines(results['by_stage'], 'Stage ')}
- **Site-specific survival**:
{survival_lines(results['by_site'])}

### Cox Proportional Hazards Model
- **C-index**: {c_index:.3f} on a 30% held-out test set ({describe_c_index(c_index)} discrimination)
- **Significant predictors** (p < 0.05; references: stage I, breast, well differentiated, female, White, no surgery/radiation/chemotherapy):
{hr_lines}

## Key Findings
1. **Stage hierarchy**: 5-year survival falls from {by_stage['I']['survival_5yr']:.0%} (stage I) to {by_stage['IV']['survival_5yr']:.0%} (stage IV)
2. **Site differences**: {worst_site} has the poorest prognosis and {best_site} the best
3. **Age effect**: Increased mortality risk with advancing age
4. **Grade and treatment**: Poorly differentiated and undifferentiated tumors carry higher risk; surgery is associated with lower risk
5. **Model performance**: {describe_c_index(c_index).capitalize()} predictive capability

## Notes
- Deaths are from any cause, so survival is lower than cancer-specific (relative) survival, especially for older patients
- Treatment effects are observational associations, not causal estimates

## Conclusions
The synthetic survival data demonstrates:
- Site- and stage-specific cancer survival patterns
- Effective risk stratification models
- Clinical relevance of key prognostic factors
- Robust analytical methodology

*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open("reports/comprehensive_survival_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    # Save summary statistics
    summary_stats.to_csv("reports/summary_statistics.csv", index=False)

    print(f"\nREPORTS GENERATED")
    print("=" * 50)
    print("SUCCESS: reports/comprehensive_survival_report.md")
    print("SUCCESS: reports/summary_statistics.csv")

    return summary_stats

if __name__ == "__main__":
    summary = generate_comprehensive_report()
    print(f"\nANALYSIS COMPLETE!")
    print("=" * 50)
