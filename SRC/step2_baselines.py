from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("midpoint_AirQuality2.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("<b>CS-4120 Midpoint Report — AirQuality2</b>", styles['Title']))
story.append(Spacer(1, 12))
story.append(Paragraph("Team: AirQuality2<br/>Dataset: Beijing PM2.5 (UCI)", styles['Normal']))
story.append(Spacer(1, 12))

# Text sections
story.append(Paragraph("<b>Updated Dataset Description</b>", styles['Heading2']))
story.append(Paragraph("43,824 hourly records ... cyclic hour/month features added.", styles['BodyText']))
story.append(Paragraph("<b>EDA Summary</b>", styles['Heading2']))
story.append(Paragraph("Classes are balanced ... correlations show TEMP and DEWP negatively correlated with PM2.5.", styles['BodyText']))

# Add plots
for plot in ["plot1_target_distribution.png","plot2_correlation_heatmap.png",
             "plot3_confusion_matrix.png","plot4_residuals_vs_predicted.png"]:
    try:
        story.append(Image(plot, width=350, height=250))
        story.append(Spacer(1,12))
    except Exception:
        pass

story.append(Paragraph("<b>Baseline Results & Discussion</b>", styles['Heading2']))
story.append(Paragraph("Logistic Regression achieved 0.77 accuracy ... Decision Tree Regressor MAE≈46 RMSE≈68.", styles['BodyText']))
story.append(Paragraph("<b>Neural Network Plan</b>", styles['Heading2']))
story.append(Paragraph("We will implement two MLPs ...", styles['BodyText']))

doc.build(story)
print("✅ midpoint_AirQuality2.pdf created")
