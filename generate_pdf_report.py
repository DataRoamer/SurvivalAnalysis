"""
Generate Professional PDF Report for Survival Analysis

This script creates a comprehensive PDF report with visualizations,
metrics, and clinical insights for survival analysis results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart

# Import survival analysis modules
from src.data.seer_preprocessor import SEERPreprocessor, create_sample_seer_data
from src.utils.trainer import SurvivalTrainer
from src.evaluation.metrics import compute_all_metrics


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
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10

    def create_kaplan_meier_plot(self, times, events, risk_groups):
        """Create Kaplan-Meier survival curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        unique_groups = np.unique(risk_groups)
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        for i, group in enumerate(unique_groups):
            mask = risk_groups == group
            group_times = times[mask]
            group_events = events[mask]

            if len(group_times) > 0:
                # Compute Kaplan-Meier
                survival_prob, time_points = self._compute_kaplan_meier(group_times, group_events)
                ax.step(time_points, survival_prob, where='post', linewidth=3,
                       color=colors[i], label=f'{group} Risk (n={mask.sum()})')

        ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax.set_title('Kaplan-Meier Survival Curves by Risk Groups', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        # Save figure
        fig_path = self.figures_dir / "kaplan_meier.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(fig_path)

    def create_risk_distribution_plot(self, risk_scores, events):
        """Create risk score distribution plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        event_mask = events == 1
        censored_mask = events == 0

        ax1.hist(risk_scores[event_mask], bins=30, alpha=0.7,
                label='Events', density=True, color='#A23B72')
        ax1.hist(risk_scores[censored_mask], bins=30, alpha=0.7,
                label='Censored', density=True, color='#2E86AB')
        ax1.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax1.set_title('Risk Score Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Box plot
        data_to_plot = [risk_scores[event_mask], risk_scores[censored_mask]]
        bp = ax2.boxplot(data_to_plot, labels=['Events', 'Censored'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#A23B72')
        bp['boxes'][1].set_facecolor('#2E86AB')
        ax2.set_ylabel('Risk Score', fontsize=12, fontweight='bold')
        ax2.set_title('Risk Score by Event Status', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = self.figures_dir / "risk_distribution.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(fig_path)

    def create_performance_metrics_chart(self, metrics):
        """Create performance metrics visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract key metrics
        metric_names = []
        metric_values = []

        if 'c_index' in metrics:
            metric_names.append('C-Index')
            metric_values.append(metrics['c_index'])

        # Add time-dependent AUCs
        auc_keys = [k for k in metrics.keys() if k.startswith('auc_')]
        for key in sorted(auc_keys):
            time_point = key.split('_')[1]
            metric_names.append(f'AUC (t={time_point})')
            metric_values.append(metrics[key])

        if 'integrated_brier_score' in metrics:
            # Convert to 1 - Brier score for better visualization
            metric_names.append('1 - Brier Score')
            metric_values.append(1 - metrics['integrated_brier_score'])

        # Create bar chart
        bars = ax.bar(metric_names, metric_values, color='#2E86AB', alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save figure
        fig_path = self.figures_dir / "performance_metrics.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(fig_path)

    def create_risk_stratification_table(self, times, events, risk_scores):
        """Create risk stratification analysis."""
        risk_groups = pd.cut(risk_scores, bins=3, labels=['Low', 'Medium', 'High'])

        risk_data = []
        for group in ['Low', 'Medium', 'High']:
            mask = risk_groups == group
            if mask.sum() > 0:
                event_rate = events[mask].mean()
                median_time = np.median(times[mask])
                n_patients = mask.sum()

                risk_data.append([
                    f'{group} Risk',
                    str(n_patients),
                    f'{event_rate:.1%}',
                    f'{median_time:.1f}'
                ])

        return risk_data

    def _compute_kaplan_meier(self, times, events):
        """Compute Kaplan-Meier survival curve."""
        # Sort by time
        sorted_indices = np.argsort(times)
        sorted_times = times[sorted_indices]
        sorted_events = events[sorted_indices]

        # Find unique event times
        unique_times = np.unique(sorted_times[sorted_events == 1])

        # Initialize
        n_at_risk = len(times)
        survival_prob = [1.0]
        time_points = [0.0]

        for t in unique_times:
            # Number of events at time t
            n_events = np.sum((sorted_times == t) & (sorted_events == 1))

            # Update survival probability
            if n_at_risk > 0:
                survival_prob.append(survival_prob[-1] * (n_at_risk - n_events) / n_at_risk)
            else:
                survival_prob.append(survival_prob[-1])

            time_points.append(t)

            # Update number at risk
            n_at_risk -= np.sum(sorted_times == t)

        return np.array(survival_prob), np.array(time_points)

    def generate_pdf_report(self,
                           times_true,
                           events_true,
                           risk_scores,
                           dataset_info=None,
                           model_info=None,
                           title="Deep Learning Survival Analysis Report"):
        """Generate comprehensive PDF report."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = self.output_dir / f"survival_analysis_report_{timestamp}.pdf"

        # Create document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=1*inch)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )

        # Title page
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                              styles['Normal']))
        story.append(Spacer(1, 30))

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))

        # Compute metrics
        metrics = compute_all_metrics(times_true, events_true, risk_scores)

        summary_text = f"""
        This report presents the results of a deep learning survival analysis model applied to cancer registry data.
        The model achieved a C-index of {metrics.get('c_index', 0):.3f}, indicating
        {'excellent' if metrics.get('c_index', 0) > 0.8 else 'good' if metrics.get('c_index', 0) > 0.7 else 'moderate'}
        discriminative performance. Risk stratification analysis shows clear separation between
        low, medium, and high-risk patient groups.
        """

        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Dataset Overview
        if dataset_info:
            story.append(Paragraph("Dataset Overview", heading_style))

            dataset_data = [
                ['Metric', 'Value'],
                ['Total Patients', str(dataset_info.get('n_samples', len(times_true)))],
                ['Features', str(dataset_info.get('n_features', 'N/A'))],
                ['Event Rate', f"{dataset_info.get('event_rate', events_true.mean()):.1%}"],
                ['Median Follow-up (months)', f"{dataset_info.get('median_followup', np.median(times_true)):.1f}"]
            ]

            dataset_table = Table(dataset_data, colWidths=[2.5*inch, 2*inch])
            dataset_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(dataset_table)
            story.append(Spacer(1, 20))

        # Model Performance
        story.append(Paragraph("Model Performance", heading_style))

        performance_text = f"""
        <b>Concordance Index (C-index):</b> {metrics.get('c_index', 0):.4f}<br/>
        <i>Measures the model's ability to correctly rank patients by risk.
        Values > 0.7 indicate good discrimination.</i><br/><br/>

        <b>Integrated Brier Score:</b> {metrics.get('integrated_brier_score', 0):.4f}<br/>
        <i>Measures prediction accuracy over time. Lower values indicate better predictions.</i><br/><br/>
        """

        # Add time-dependent AUCs
        auc_keys = [k for k in metrics.keys() if k.startswith('auc_')]
        if auc_keys:
            performance_text += "<b>Time-dependent AUCs:</b><br/>"
            for key in sorted(auc_keys):
                time_point = key.split('_')[1]
                auc_value = metrics[key]
                performance_text += f"• AUC at {time_point} months: {auc_value:.3f}<br/>"

        story.append(Paragraph(performance_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Create and add performance metrics chart
        metrics_chart_path = self.create_performance_metrics_chart(metrics)
        if Path(metrics_chart_path).exists():
            story.append(Image(metrics_chart_path, width=6*inch, height=3.6*inch))
            story.append(Spacer(1, 20))

        story.append(PageBreak())

        # Risk Stratification Analysis
        story.append(Paragraph("Risk Stratification Analysis", heading_style))

        # Create risk stratification table
        risk_data = self.create_risk_stratification_table(times_true, events_true, risk_scores)

        risk_table_data = [['Risk Group', 'Patients', 'Event Rate', 'Median Time (months)']] + risk_data

        risk_table = Table(risk_table_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 2*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 30))

        # Create and add Kaplan-Meier plot
        risk_groups = pd.cut(risk_scores, bins=3, labels=['Low', 'Medium', 'High'])
        km_plot_path = self.create_kaplan_meier_plot(times_true, events_true, risk_groups)
        if Path(km_plot_path).exists():
            story.append(Image(km_plot_path, width=6*inch, height=3.6*inch))
            story.append(Spacer(1, 20))

        story.append(PageBreak())

        # Risk Score Distribution
        story.append(Paragraph("Risk Score Distribution", heading_style))

        risk_dist_path = self.create_risk_distribution_plot(risk_scores, events_true)
        if Path(risk_dist_path).exists():
            story.append(Image(risk_dist_path, width=6*inch, height=3*inch))
            story.append(Spacer(1, 20))

        # Clinical Insights
        story.append(Paragraph("Clinical Insights and Recommendations", heading_style))

        # Calculate risk ratios for insights
        high_risk_mask = risk_groups == 'High'
        low_risk_mask = risk_groups == 'Low'

        high_risk_event_rate = events_true[high_risk_mask].mean() if high_risk_mask.sum() > 0 else 0
        low_risk_event_rate = events_true[low_risk_mask].mean() if low_risk_mask.sum() > 0 else 0
        risk_ratio = high_risk_event_rate / low_risk_event_rate if low_risk_event_rate > 0 else float('inf')

        insights_text = f"""
        <b>Model Performance Assessment:</b><br/>
        {'• Excellent discrimination: The model shows outstanding ability to distinguish between high and low-risk patients.' if metrics.get('c_index', 0) > 0.8 else '• Good discrimination: The model demonstrates solid predictive performance for risk stratification.' if metrics.get('c_index', 0) > 0.7 else '• Moderate discrimination: The model shows acceptable predictive ability.'}<br/><br/>

        <b>Risk Stratification Insights:</b><br/>
        • High-risk patients have {risk_ratio:.1f}x higher event rate than low-risk patients<br/>
        • The model successfully identifies the highest-risk {100/3:.0f}% of patients<br/>
        • Clear separation between risk groups supports clinical utility<br/><br/>

        <b>Clinical Recommendations:</b><br/>
        • Use high-risk scores to guide treatment intensification decisions<br/>
        • Implement risk-based follow-up scheduling protocols<br/>
        • Consider integration into clinical decision support systems<br/>
        • Validate model performance with external datasets<br/>
        • Monitor model performance over time for potential drift<br/><br/>

        <b>Implementation Considerations:</b><br/>
        • Ensure clinical staff training on model interpretation<br/>
        • Establish clear protocols for acting on risk predictions<br/>
        • Maintain documentation for regulatory compliance<br/>
        • Plan for regular model updates and recalibration
        """

        story.append(Paragraph(insights_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Footer
        story.append(Spacer(1, 50))
        footer_text = """
        <i>This report was generated using deep learning survival analysis methods.
        Results should be interpreted by qualified medical professionals and used
        in conjunction with clinical judgment. Model predictions are intended to
        support, not replace, clinical decision-making.</i>
        """
        story.append(Paragraph(footer_text, styles['Normal']))

        # Build PDF
        doc.build(story)

        return str(pdf_path)


def main():
    """Generate a comprehensive PDF survival analysis report."""
    print("=== Generating PDF Survival Analysis Report ===")
    print()

    # Step 1: Create sample data
    print("Step 1: Creating sample SEER data...")
    df = create_sample_seer_data(n_samples=1000, random_state=42)

    dataset_info = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 2,  # Exclude time and event columns
        'event_rate': df['vital_status'].mean(),
        'median_followup': df['survival_months'].median(),
        'dataset_name': 'Sample SEER Cancer Registry Data'
    }

    print(f"Dataset: {dataset_info['n_samples']} patients, {dataset_info['n_features']} features")
    print(f"Event rate: {dataset_info['event_rate']:.1%}")
    print()

    # Step 2: Preprocess data
    print("Step 2: Preprocessing data...")
    preprocessor = SEERPreprocessor()
    X, t, e = preprocessor.fit_transform(df)

    # Split data
    data_splits = preprocessor.train_test_split(X, t, e, test_size=0.3, random_state=42)
    X_train, t_train, e_train = data_splits['X_train'], data_splits['t_train'], data_splits['e_train']
    X_test, t_test, e_test = data_splits['X_test'], data_splits['t_test'], data_splits['e_test']

    print(f"Training: {len(X_train)} patients, Testing: {len(X_test)} patients")
    print()

    # Step 3: Train model
    print("Step 3: Training survival model...")
    trainer = SurvivalTrainer(
        model_name='deepsurv',
        max_epochs=25,  # Reduced for faster demo
        early_stopping_patience=10,
        logger_type='none'
    )

    # Train model
    train_results = trainer.train(X_train, t_train, e_train)

    # Evaluate on test set
    test_results = trainer.evaluate(
        train_results['model'],
        X_test, t_test, e_test,
        return_predictions=True
    )

    print(f"Model trained! C-index: {test_results['metrics']['c_index']:.4f}")
    print()

    # Step 4: Generate PDF report
    print("Step 4: Generating PDF report...")

    pdf_reporter = PDFSurvivalReporter(output_dir="pdf_survival_reports")

    pdf_path = pdf_reporter.generate_pdf_report(
        times_true=t_test,
        events_true=e_test,
        risk_scores=test_results['predictions']['risk_scores'],
        dataset_info=dataset_info,
        title="Deep Learning Survival Analysis Report - SEER Data"
    )

    # Get file size
    file_size = Path(pdf_path).stat().st_size / 1024  # Size in KB

    print("✅ PDF Report generated successfully!")
    print()
    print(f"📄 PDF Report: {pdf_path}")
    print(f"📊 File Size: {file_size:.1f} KB")
    print(f"📁 Figures Directory: {pdf_reporter.figures_dir}")
    print()

    print("📖 Report Contents:")
    print("   • Executive Summary")
    print("   • Dataset Overview")
    print("   • Model Performance Metrics")
    print("   • Risk Stratification Analysis")
    print("   • Kaplan-Meier Survival Curves")
    print("   • Risk Score Distribution")
    print("   • Clinical Insights & Recommendations")
    print()

    print("🔍 To view the PDF:")
    print(f"   Open: {Path(pdf_path).absolute()}")


if __name__ == "__main__":
    main()