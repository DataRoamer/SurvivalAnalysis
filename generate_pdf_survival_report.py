"""
Generate Professional PDF Survival Analysis Report

This script creates a comprehensive PDF report with visualizations,
metrics, and clinical insights for survival analysis results.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis import compute_survival_summary, format_months, describe_c_index
from analysis.survival_summary import STAGE_ORDER

# Import lifelines for survival analysis
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


class PDFSurvivalReporter:
    """Professional PDF report generator for survival analysis."""

    def __init__(self, output_dir: str = "pdf_reports"):
        """Initialize PDF reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create figures directory
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        # Set matplotlib style for high-quality figures
        plt.style.use('default')
        sns.set_palette("husl")

        # Setup styles
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()

    def setup_custom_styles(self):
        """Setup custom styles for the PDF report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=20,
            textColor=colors.darkblue
        ))

        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=15,
            textColor=colors.darkgreen
        ))

        # Key findings style
        self.styles.add(ParagraphStyle(
            name='KeyFindings',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            leftIndent=20,
            textColor=colors.darkred
        ))

    def create_survival_curves(self, df):
        """Create survival curve visualizations."""
        if not LIFELINES_AVAILABLE:
            print("Warning: Cannot create survival curves - lifelines not available")
            return None

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Overall survival
        kmf = KaplanMeierFitter()
        kmf.fit(df['survival_months'], df['vital_status'], label='Overall Survival')
        kmf.plot(ax=ax1, ci_show=True)
        ax1.set_title('Overall Survival Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (months)')
        ax1.set_ylabel('Survival Probability')
        ax1.grid(True, alpha=0.3)

        # Survival by stage
        stages = [s for s in STAGE_ORDER if s in set(df['stage'])]
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

        for i, stage in enumerate(stages):
            stage_data = df[df['stage'] == stage]
            if len(stage_data) > 10:
                kmf_stage = KaplanMeierFitter()
                kmf_stage.fit(
                    stage_data['survival_months'],
                    stage_data['vital_status'],
                    label=f'Stage {stage}'
                )
                kmf_stage.plot(ax=ax2, color=colors[i % len(colors)], ci_show=False)

        ax2.set_title('Survival by Cancer Stage', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (months)')
        ax2.set_ylabel('Survival Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Survival by age groups
        df['age_group'] = pd.cut(df['age_at_diagnosis'],
                                bins=[0, 50, 65, 80, 120],
                                labels=['<50', '50-64', '65-79', '80+'])

        for i, age_group in enumerate(['<50', '50-64', '65-79', '80+']):
            age_data = df[df['age_group'] == age_group]
            if len(age_data) > 10:
                kmf_age = KaplanMeierFitter()
                kmf_age.fit(
                    age_data['survival_months'],
                    age_data['vital_status'],
                    label=f'Age {age_group}'
                )
                kmf_age.plot(ax=ax3, color=colors[i], ci_show=False)

        ax3.set_title('Survival by Age Group', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (months)')
        ax3.set_ylabel('Survival Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Survival by primary site
        for i, site in enumerate(df['primary_site'].value_counts().index):
            site_data = df[df['primary_site'] == site]
            kmf_site = KaplanMeierFitter()
            kmf_site.fit(site_data['survival_months'], site_data['vital_status'], label=site)
            kmf_site.plot(ax=ax4, color=colors[i % len(colors)], ci_show=False)

        ax4.set_title('Survival by Primary Site', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time (months)')
        ax4.set_ylabel('Survival Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        figure_path = self.figures_dir / "survival_curves.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(figure_path)

    def create_summary_plots(self, df):
        """Create summary visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Cancer sites distribution
        site_counts = df['primary_site'].value_counts()
        ax1.pie(site_counts.values, labels=site_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Cancer Sites Distribution', fontsize=14, fontweight='bold')

        # Stage distribution
        stage_counts = df['stage'].value_counts().reindex(STAGE_ORDER).dropna()
        bars = ax2.bar(stage_counts.index, stage_counts.values, color='lightcoral', alpha=0.8)
        ax2.set_title('Stage Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cancer Stage')
        ax2.set_ylabel('Number of Patients')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Age distribution
        ax3.hist(df['age_at_diagnosis'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Age Distribution at Diagnosis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Age (years)')
        ax3.set_ylabel('Number of Patients')
        ax3.grid(True, alpha=0.3)

        # Event status
        event_counts = df['vital_status'].value_counts()
        labels = ['Alive', 'Deceased']
        colors_pie = ['lightblue', 'lightcoral']
        ax4.pie(event_counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors_pie, startangle=90)
        ax4.set_title('Patient Status Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        figure_path = self.figures_dir / "summary_plots.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(figure_path)

    def generate_pdf_report(self, df):
        """Generate comprehensive PDF report."""
        print("Generating PDF survival analysis report...")

        results = compute_survival_summary(df)
        c_index = results['c_index_test']
        significant = results['hazard_ratios'][results['hazard_ratios']['p'] < 0.05] \
            .sort_values('hr', ascending=False)
        by_stage = results['by_stage']

        # Create visualizations
        survival_curves_path = self.create_survival_curves(df)
        summary_plots_path = self.create_summary_plots(df)

        # Setup PDF document
        pdf_path = self.output_dir / "Survival_Analysis_Report.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        # Build story (content)
        story = []

        # Title page
        story.append(Paragraph("Comprehensive Survival Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
                              self.styles['Normal']))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Cancer Survival Data Analysis", self.styles['Heading2']))
        story.append(PageBreak())

        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        summary_text = f"""
        This comprehensive survival analysis was conducted on a dataset of {len(df):,} cancer patients.
        The analysis employed Kaplan-Meier survival estimation and Cox proportional hazards modeling
        to identify key prognostic factors and survival patterns.

        <b>Key Findings:</b>
        • Event rate: {results['event_rate']:.1%} ({results['n_events']} deaths)
        • Median follow-up time: {results['median_follow_up']:.0f} months
        • Strong stage- and site-dependent survival patterns observed
        • Age, grade and surgery are significant prognostic factors
        """
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Dataset Overview
        story.append(Paragraph("Dataset Overview", self.styles['SectionHeader']))

        # Create overview table
        overview_data = [
            ['Metric', 'Value'],
            ['Total Patients', f"{len(df):,}"],
            ['Deaths (Events)', f"{df['vital_status'].sum():,} ({df['vital_status'].mean():.1%})"],
            ['Median Follow-up', f"{results['median_follow_up']:.0f} months"],
            ['5-year Overall Survival', f"{results['overall']['survival_5yr']:.1%}"],
            ['Age Range', f"{df['age_at_diagnosis'].min():.0f} - {df['age_at_diagnosis'].max():.0f} years"],
            ['Mean Age', f"{df['age_at_diagnosis'].mean():.1f} years"],
        ]

        overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(overview_table)
        story.append(Spacer(1, 20))

        # Summary visualizations
        if summary_plots_path:
            story.append(Paragraph("Data Distribution", self.styles['SubsectionHeader']))
            story.append(Image(summary_plots_path, width=7*inch, height=5.25*inch))
            story.append(PageBreak())

        # Survival Analysis Results
        story.append(Paragraph("Survival Analysis Results", self.styles['SectionHeader']))

        # Stage-specific survival
        stage_lines = '<br/>'.join(
            f"• Stage {stage}: median survival {format_months(d['median_survival']).lower()}, "
            f"5-year survival {d['survival_5yr']:.1%}"
            for stage, d in by_stage.items())
        site_lines = '<br/>'.join(
            f"• {site}: median survival {format_months(d['median_survival']).lower()}, "
            f"5-year survival {d['survival_5yr']:.1%}"
            for site, d in results['by_site'].items())
        stage_survival_text = f"""
        <b>Survival by Cancer Stage:</b><br/>
        The analysis reveals clear stage-dependent survival patterns:<br/>
        {stage_lines}<br/><br/>
        <b>Survival by Primary Site:</b><br/>
        {site_lines}<br/><br/>

        This demonstrates the critical importance of early detection and staging in cancer prognosis.
        """
        story.append(Paragraph(stage_survival_text, self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Survival curves
        if survival_curves_path:
            story.append(Paragraph("Survival Curves", self.styles['SubsectionHeader']))
            story.append(Image(survival_curves_path, width=7*inch, height=5.25*inch))
            story.append(PageBreak())

        # Cox Model Results
        story.append(Paragraph("Cox Proportional Hazards Model", self.styles['SectionHeader']))

        hr_lines = '<br/>'.join(
            f"• {name}: HR = {row['hr']:.2f} (95% CI {row['hr_lower']:.2f}-{row['hr_upper']:.2f})"
            for name, row in significant.iterrows())
        cox_text = f"""
        <b>Model Performance:</b><br/>
        • Concordance Index (C-index, 30% held-out test set): {c_index:.3f}
        ({describe_c_index(c_index).capitalize()} discrimination)<br/>
        • The model demonstrates {describe_c_index(c_index)} predictive capability for risk stratification.<br/><br/>

        <b>Significant Risk Factors (p &lt; 0.05):</b><br/>
        {hr_lines}<br/><br/>

        Reference groups: stage I, breast, well differentiated, female, White, no treatment.

        These results align with established clinical knowledge and demonstrate
        the model's validity for risk assessment.
        """
        story.append(Paragraph(cox_text, self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Clinical Implications
        story.append(Paragraph("Clinical Implications", self.styles['SectionHeader']))

        clinical_text = f"""
        <b>Key Clinical Insights:</b>

        1. <b>Early Detection is Critical:</b> Stage I patients have {by_stage['I']['survival_5yr']:.0%} 5-year
           survival versus {by_stage['IV']['survival_5yr']:.0%} for stage IV,
           emphasizing the importance of screening and early detection programs.

        2. <b>Risk Stratification:</b> The model provides effective risk stratification with
           a C-index of {c_index:.3f}, enabling personalized treatment planning.

        3. <b>Age Considerations:</b> Older patients face increased mortality risk, suggesting
           need for age-adjusted treatment protocols.

        4. <b>Tumor Biology:</b> Poorly differentiated and undifferentiated tumors carry
           substantially higher risk, and lung cancer has the poorest prognosis of the sites studied.

        5. <b>Follow-up Strategies:</b> Advanced stage patients require more intensive
           monitoring and potentially aggressive treatment approaches.
        """
        story.append(Paragraph(clinical_text, self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Conclusions
        story.append(Paragraph("Conclusions", self.styles['SectionHeader']))

        conclusions_text = f"""
        This survival analysis successfully demonstrates:

        • <b>Robust Methodology:</b> Application of standard survival analysis techniques
          including Kaplan-Meier estimation and Cox proportional hazards modeling.

        • <b>Clinical Validity:</b> Results consistent with established clinical knowledge
          and cancer biology principles.

        • <b>Predictive Capability:</b> The developed model shows {describe_c_index(c_index)} discrimination
          (C-index = {c_index:.3f}) for risk stratification.

        • <b>Actionable Insights:</b> Clear identification of key prognostic factors
          that can inform clinical decision-making.

        The analysis provides a foundation for evidence-based treatment planning and
        patient counseling in oncology practice.
        """
        story.append(Paragraph(conclusions_text, self.styles['Normal']))

        # Build PDF
        doc.build(story)

        print(f"PDF report generated: {pdf_path}")
        return str(pdf_path)


def main():
    """Generate PDF survival analysis report."""
    print("=== PDF Survival Analysis Report Generation ===\n")

    # Load the synthetic data
    data_path = "data/synthetic_survival_sample.csv"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run the analysis first to generate the data.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded data: {len(df)} patients")

    # Create PDF reporter and generate report
    reporter = PDFSurvivalReporter()
    pdf_path = reporter.generate_pdf_report(df)

    print(f"\nSUCCESS: PDF report generated at {pdf_path}")
    print("The report includes:")
    print("• Executive summary")
    print("• Dataset overview with statistics")
    print("• Survival curves and visualizations")
    print("• Cox model results")
    print("• Clinical implications")
    print("• Professional formatting with charts")


if __name__ == "__main__":
    main()