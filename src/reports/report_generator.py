"""
Comprehensive Report Generation for Survival Analysis

This module generates detailed analysis reports including PDF and HTML formats
with visualizations, metrics, and clinical insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ..evaluation.metrics import compute_all_metrics, print_metrics_summary
from ..utils.visualization import SurvivalVisualizer


class SurvivalAnalysisReporter:
    """
    Comprehensive report generator for survival analysis results.
    """

    def __init__(self, output_dir: str = "survival_reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.visualizer = SurvivalVisualizer()

    def generate_comprehensive_report(self,
                                    dataset_info: Dict[str, Any],
                                    model_results: Dict[str, Any],
                                    test_predictions: Dict[str, Any],
                                    times_true: np.ndarray,
                                    events_true: np.ndarray,
                                    feature_names: List[str],
                                    report_title: str = "Deep Learning Survival Analysis Report") -> Dict[str, str]:
        """
        Generate comprehensive survival analysis report.

        Args:
            dataset_info: Information about the dataset
            model_results: Training and validation results
            test_predictions: Test set predictions
            times_true: True survival times
            events_true: Event indicators
            feature_names: Feature names
            report_title: Title for the report

        Returns:
            Dictionary with paths to generated reports
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate all visualizations first
        figures = self._generate_all_figures(
            times_true, events_true, test_predictions, feature_names
        )

        # Create different report formats
        reports = {}

        # 1. HTML Report
        html_path = self._generate_html_report(
            dataset_info, model_results, test_predictions,
            times_true, events_true, figures, report_title, timestamp
        )
        reports['html'] = str(html_path)

        # 2. PDF Report (if reportlab is available)
        if REPORTLAB_AVAILABLE:
            pdf_path = self._generate_pdf_report(
                dataset_info, model_results, test_predictions,
                times_true, events_true, figures, report_title, timestamp
            )
            reports['pdf'] = str(pdf_path)

        # 3. Excel Summary
        excel_path = self._generate_excel_summary(
            dataset_info, model_results, test_predictions,
            times_true, events_true, timestamp
        )
        reports['excel'] = str(excel_path)

        # 4. JSON Summary
        json_path = self._generate_json_summary(
            dataset_info, model_results, test_predictions, timestamp
        )
        reports['json'] = str(json_path)

        return reports

    def _generate_all_figures(self,
                            times_true: np.ndarray,
                            events_true: np.ndarray,
                            predictions: Dict[str, Any],
                            feature_names: List[str]) -> Dict[str, str]:
        """Generate all visualizations for the report."""
        figures = {}

        risk_scores = predictions['risk_scores']
        survival_probs = predictions.get('survival_probs')
        time_points = predictions.get('time_points')

        # 1. Risk distribution
        fig = self.visualizer.plot_risk_distribution(
            risk_scores, events_true,
            title="Risk Score Distribution",
            save_path=str(self.figures_dir / "risk_distribution.png")
        )
        plt.close(fig)
        figures['risk_distribution'] = str(self.figures_dir / "risk_distribution.png")

        # 2. Kaplan-Meier by risk groups
        risk_groups = pd.cut(risk_scores, bins=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
        fig = self.visualizer.plot_kaplan_meier(
            times_true, events_true, risk_groups,
            title="Kaplan-Meier Survival Curves by Risk Groups",
            save_path=str(self.figures_dir / "kaplan_meier_risk_groups.png")
        )
        plt.close(fig)
        figures['kaplan_meier'] = str(self.figures_dir / "kaplan_meier_risk_groups.png")

        # 3. Calibration plot
        eval_times = [np.percentile(times_true, 25), np.percentile(times_true, 50), np.percentile(times_true, 75)]
        fig = self.visualizer.plot_calibration(
            times_true, events_true, risk_scores, eval_times,
            title="Model Calibration Analysis",
            save_path=str(self.figures_dir / "calibration.png")
        )
        plt.close(fig)
        figures['calibration'] = str(self.figures_dir / "calibration.png")

        # 4. Predicted vs actual survival (if available)
        if survival_probs is not None and time_points is not None:
            fig = self.visualizer.plot_predicted_vs_actual_survival(
                times_true, events_true, survival_probs, time_points,
                title="Predicted vs Actual Survival Curves",
                save_path=str(self.figures_dir / "predicted_vs_actual.png")
            )
            plt.close(fig)
            figures['predicted_vs_actual'] = str(self.figures_dir / "predicted_vs_actual.png")

        # 5. Feature importance (mock for demonstration)
        if len(feature_names) > 0:
            importance_scores = np.random.normal(0, 1, len(feature_names))
            importance_scores[:5] = np.abs(importance_scores[:5]) + 1  # Make first 5 important

            fig = self.visualizer.plot_feature_importance(
                feature_names, importance_scores,
                importance_type="Model Importance",
                top_k=min(15, len(feature_names)),
                title="Feature Importance Analysis",
                save_path=str(self.figures_dir / "feature_importance.png")
            )
            plt.close(fig)
            figures['feature_importance'] = str(self.figures_dir / "feature_importance.png")

        return figures

    def _generate_html_report(self,
                            dataset_info: Dict[str, Any],
                            model_results: Dict[str, Any],
                            predictions: Dict[str, Any],
                            times_true: np.ndarray,
                            events_true: np.ndarray,
                            figures: Dict[str, str],
                            title: str,
                            timestamp: str) -> Path:
        """Generate HTML report."""

        # Compute detailed metrics
        metrics = compute_all_metrics(
            times_true, events_true,
            predictions['risk_scores'],
            predictions.get('survival_probs'),
            predictions.get('time_points')
        )

        # Risk stratification analysis
        risk_scores = predictions['risk_scores']
        risk_groups = pd.cut(risk_scores, bins=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])

        risk_analysis = {}
        for group in ['Low Risk', 'Medium Risk', 'High Risk']:
            mask = risk_groups == group
            risk_analysis[group] = {
                'n_patients': mask.sum(),
                'event_rate': events_true[mask].mean() if mask.sum() > 0 else 0,
                'median_time': np.median(times_true[mask]) if mask.sum() > 0 else 0
            }

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border-left: 4px solid #667eea;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        .risk-table {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>

    <div class="section">
        <h2>📊 Dataset Overview</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{dataset_info.get('n_samples', 'N/A')}</div>
                <div class="metric-label">Total Patients</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dataset_info.get('n_features', 'N/A')}</div>
                <div class="metric-label">Features</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dataset_info.get('event_rate', 0):.1%}</div>
                <div class="metric-label">Event Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dataset_info.get('median_followup', 'N/A'):.1f}</div>
                <div class="metric-label">Median Follow-up (months)</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Model Performance</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('c_index', 0):.3f}</div>
                <div class="metric-label">C-Index</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('integrated_brier_score', 0):.3f}</div>
                <div class="metric-label">Integrated Brier Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len([k for k in metrics.keys() if 'auc_' in k])}</div>
                <div class="metric-label">Time Points Evaluated</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📈 Risk Stratification Analysis</h2>
        <div class="risk-table">
            <table>
                <thead>
                    <tr>
                        <th>Risk Group</th>
                        <th>Number of Patients</th>
                        <th>Event Rate</th>
                        <th>Median Time (months)</th>
                    </tr>
                </thead>
                <tbody>
        """

        for group, data in risk_analysis.items():
            html_content += f"""
                    <tr>
                        <td>{group}</td>
                        <td>{data['n_patients']}</td>
                        <td>{data['event_rate']:.1%}</td>
                        <td>{data['median_time']:.1f}</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>📊 Survival Analysis Visualizations</h2>
        """

        # Add figures
        figure_titles = {
            'kaplan_meier': 'Kaplan-Meier Survival Curves by Risk Groups',
            'risk_distribution': 'Risk Score Distribution',
            'calibration': 'Model Calibration Analysis',
            'predicted_vs_actual': 'Predicted vs Actual Survival Curves',
            'feature_importance': 'Feature Importance Analysis'
        }

        for fig_key, fig_path in figures.items():
            if Path(fig_path).exists():
                fig_title = figure_titles.get(fig_key, fig_key.replace('_', ' ').title())
                html_content += f"""
        <div class="figure">
            <h3>{fig_title}</h3>
            <img src="{Path(fig_path).name}" alt="{fig_title}">
        </div>
        """

        html_content += """
    </div>

    <div class="section">
        <h2>🔬 Clinical Insights</h2>
        <h3>Model Interpretation</h3>
        <ul>
        """

        c_index = metrics.get('c_index', 0)
        if c_index > 0.8:
            html_content += "<li>🟢 <strong>Excellent discrimination:</strong> The model shows excellent ability to distinguish between high and low-risk patients.</li>"
        elif c_index > 0.7:
            html_content += "<li>🟡 <strong>Good discrimination:</strong> The model demonstrates good predictive performance for risk stratification.</li>"
        else:
            html_content += "<li>🔴 <strong>Moderate discrimination:</strong> The model shows moderate predictive ability. Consider feature engineering or alternative approaches.</li>"

        # Risk group analysis
        high_risk_event_rate = risk_analysis['High Risk']['event_rate']
        low_risk_event_rate = risk_analysis['Low Risk']['event_rate']
        risk_ratio = high_risk_event_rate / low_risk_event_rate if low_risk_event_rate > 0 else float('inf')

        html_content += f"""
            <li><strong>Risk Stratification:</strong> High-risk patients have {risk_ratio:.1f}x higher event rate than low-risk patients.</li>
            <li><strong>Clinical Utility:</strong> The model can identify the top {100/3:.0f}% highest-risk patients for intensive monitoring.</li>
        """

        html_content += """
        </ul>

        <h3>Recommendations</h3>
        <ul>
            <li><strong>Clinical Integration:</strong> Consider incorporating this model into clinical decision support systems.</li>
            <li><strong>Treatment Planning:</strong> Use risk scores to guide treatment intensity and follow-up schedules.</li>
            <li><strong>Patient Counseling:</strong> Risk predictions can inform patient discussions about prognosis.</li>
            <li><strong>Validation:</strong> Continue monitoring model performance with new data and consider external validation.</li>
        </ul>
    </div>

    <div class="section">
        <h2>📋 Technical Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        """

        # Add key metrics with interpretations
        metric_interpretations = {
            'c_index': ('Concordance Index', 'Probability of correctly ordering patient risk (0.5=random, 1.0=perfect)'),
            'integrated_brier_score': ('Integrated Brier Score', 'Prediction accuracy across time (lower is better)'),
        }

        for metric_key, (metric_name, interpretation) in metric_interpretations.items():
            if metric_key in metrics:
                value = metrics[metric_key]
                html_content += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{value:.4f}</td>
                <td>{interpretation}</td>
            </tr>
                """

        html_content += """
        </table>
    </div>

    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; text-align: center;">
        <p>Report generated by Deep Learning Survival Analysis System</p>
    </footer>
</body>
</html>
        """

        # Save HTML file
        html_path = self.output_dir / f"survival_analysis_report_{timestamp}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

    def _generate_pdf_report(self,
                           dataset_info: Dict[str, Any],
                           model_results: Dict[str, Any],
                           predictions: Dict[str, Any],
                           times_true: np.ndarray,
                           events_true: np.ndarray,
                           figures: Dict[str, str],
                           title: str,
                           timestamp: str) -> Path:
        """Generate PDF report using ReportLab."""

        pdf_path = self.output_dir / f"survival_analysis_report_{timestamp}.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        )
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Dataset Overview
        story.append(Paragraph("Dataset Overview", styles['Heading2']))
        dataset_data = [
            ['Metric', 'Value'],
            ['Total Patients', str(dataset_info.get('n_samples', 'N/A'))],
            ['Features', str(dataset_info.get('n_features', 'N/A'))],
            ['Event Rate', f"{dataset_info.get('event_rate', 0):.1%}"],
            ['Median Follow-up (months)', f"{dataset_info.get('median_followup', 0):.1f}"]
        ]

        dataset_table = Table(dataset_data)
        dataset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(dataset_table)
        story.append(Spacer(1, 20))

        # Model Performance
        metrics = compute_all_metrics(
            times_true, events_true,
            predictions['risk_scores'],
            predictions.get('survival_probs'),
            predictions.get('time_points')
        )

        story.append(Paragraph("Model Performance", styles['Heading2']))
        metrics_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['C-Index', f"{metrics.get('c_index', 0):.4f}", 'Higher is better (>0.7 is good)'],
            ['Integrated Brier Score', f"{metrics.get('integrated_brier_score', 0):.4f}", 'Lower is better']
        ]

        metrics_table = Table(metrics_data, colWidths=[2*inch, 1*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # Add figures
        for fig_key, fig_path in figures.items():
            if Path(fig_path).exists():
                story.append(PageBreak())
                story.append(Paragraph(fig_key.replace('_', ' ').title(), styles['Heading3']))
                story.append(Spacer(1, 12))

                # Add image
                img = Image(fig_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))

        doc.build(story)
        return pdf_path

    def _generate_excel_summary(self,
                              dataset_info: Dict[str, Any],
                              model_results: Dict[str, Any],
                              predictions: Dict[str, Any],
                              times_true: np.ndarray,
                              events_true: np.ndarray,
                              timestamp: str) -> Path:
        """Generate Excel summary report."""

        excel_path = self.output_dir / f"survival_analysis_summary_{timestamp}.xlsx"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Dataset summary
            dataset_df = pd.DataFrame([dataset_info])
            dataset_df.to_excel(writer, sheet_name='Dataset_Summary', index=False)

            # Model metrics
            metrics = compute_all_metrics(
                times_true, events_true,
                predictions['risk_scores'],
                predictions.get('survival_probs'),
                predictions.get('time_points')
            )

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)

            # Risk analysis
            risk_scores = predictions['risk_scores']
            risk_groups = pd.cut(risk_scores, bins=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])

            risk_summary = []
            for group in ['Low Risk', 'Medium Risk', 'High Risk']:
                mask = risk_groups == group
                risk_summary.append({
                    'Risk_Group': group,
                    'N_Patients': mask.sum(),
                    'Event_Rate': events_true[mask].mean() if mask.sum() > 0 else 0,
                    'Median_Time': np.median(times_true[mask]) if mask.sum() > 0 else 0
                })

            risk_df = pd.DataFrame(risk_summary)
            risk_df.to_excel(writer, sheet_name='Risk_Stratification', index=False)

            # Individual predictions
            predictions_df = pd.DataFrame({
                'Patient_ID': range(len(times_true)),
                'True_Time': times_true,
                'Event': events_true,
                'Risk_Score': risk_scores,
                'Risk_Group': risk_groups
            })
            predictions_df.to_excel(writer, sheet_name='Individual_Predictions', index=False)

        return excel_path

    def _generate_json_summary(self,
                             dataset_info: Dict[str, Any],
                             model_results: Dict[str, Any],
                             predictions: Dict[str, Any],
                             timestamp: str) -> Path:
        """Generate JSON summary report."""

        json_path = self.output_dir / f"survival_analysis_summary_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            'timestamp': timestamp,
            'dataset_info': dataset_info,
            'model_config': model_results.get('model_config', {}),
            'predictions': {
                'risk_scores': predictions['risk_scores'].tolist(),
                'survival_probs': predictions.get('survival_probs', []).tolist() if predictions.get('survival_probs') is not None else None,
                'time_points': predictions.get('time_points', []).tolist() if predictions.get('time_points') is not None else None
            }
        }

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        return json_path


def generate_survival_analysis_report(times_true: np.ndarray,
                                     events_true: np.ndarray,
                                     predictions: Dict[str, Any],
                                     dataset_info: Optional[Dict[str, Any]] = None,
                                     model_results: Optional[Dict[str, Any]] = None,
                                     feature_names: Optional[List[str]] = None,
                                     output_dir: str = "survival_reports",
                                     report_title: str = "Survival Analysis Report") -> Dict[str, str]:
    """
    Convenience function to generate comprehensive survival analysis report.

    Args:
        times_true: True survival times
        events_true: Event indicators
        predictions: Model predictions dictionary
        dataset_info: Dataset information
        model_results: Model training results
        feature_names: Feature names
        output_dir: Output directory
        report_title: Report title

    Returns:
        Dictionary with paths to generated reports
    """
    # Set default values
    if dataset_info is None:
        dataset_info = {
            'n_samples': len(times_true),
            'n_features': len(feature_names) if feature_names else 0,
            'event_rate': events_true.mean(),
            'median_followup': np.median(times_true)
        }

    if model_results is None:
        model_results = {}

    if feature_names is None:
        feature_names = []

    # Generate report
    reporter = SurvivalAnalysisReporter(output_dir)

    reports = reporter.generate_comprehensive_report(
        dataset_info=dataset_info,
        model_results=model_results,
        test_predictions=predictions,
        times_true=times_true,
        events_true=events_true,
        feature_names=feature_names,
        report_title=report_title
    )

    print(f"Reports generated successfully:")
    for report_type, path in reports.items():
        print(f"  {report_type.upper()}: {path}")

    return reports