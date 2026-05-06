"""
PDF Report Generator for Traffic Violations
Generates professional PDF reports with images and AI analysis
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from pathlib import Path
import cv2
from typing import Dict, List, Any


class ViolationPDFGenerator:
    """Generate professional PDF reports for traffic violations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_violation_report(
        self,
        car_id: int,
        violation_data: Dict[str, Any],
        plate_images: List[tuple],  # List of (image_path, confidence)
        ai_analysis: str,
        video_frame_path: str = None
    ) -> Path:
        """
        Generate a comprehensive PDF report for a single violation
        """
        filename = self.output_dir / f"violation_report_car{car_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#d32f2f'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1976d2'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        caption_style = ParagraphStyle(
            'CustomCaption',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=6
        )
        
        # Title
        story.append(Paragraph("🚨 TRAFFIC VIOLATION REPORT", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Violation Summary Box
        summary_data = [
            ['Report ID:', f'VR-{car_id}-{datetime.now().strftime("%Y%m%d")}'],
            ['Car ID:', str(car_id)],
            ['Violation Date:', violation_data.get('timestamp', 'N/A')],
            ['Traffic Light:', violation_data.get('traffic_light', 'N/A').upper()],
            ['Stop Line Crossed:', 'YES' if violation_data.get('crossed_stop_line') else 'NO'],
            ['Detection Confidence:', f"{violation_data.get('confidence', 0):.1%}"],
            ['Frame Number:', str(violation_data.get('frame_number', 'N/A'))],
        ]
        
        # Add plate number comparison if available
        if 'plate_number' in violation_data:
            plate_num = violation_data['plate_number']
            verified = violation_data.get('plate_verified', False)
            ocr_conf = violation_data.get('ocr_confidence', 0)
            ocr_raw = violation_data.get('ocr_raw', '')
            
            status = f"✓ Verified ({ocr_conf:.0%})" if verified else "⚠ Unverified"
            
            # Show raw OCR vs OpenAI result
            if ocr_raw and ocr_raw != plate_num:
                summary_data.append(['Raw OCR Text:', ocr_raw])
                summary_data.append(['OpenAI Verified:', f"{plate_num} - {status}"])
            else:
                summary_data.append(['License Plate:', f"{plate_num} - {status}"])
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#90caf9')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.4*inch))
        
        # AI Analysis Section
        story.append(Paragraph("🤖 AI VIOLATION ANALYSIS", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        analysis_style = ParagraphStyle(
            'Analysis',
            parent=styles['BodyText'],
            fontSize=10,
            leading=14,
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=20,
            rightIndent=20
        )
        story.append(Paragraph(ai_analysis.replace('\n', '<br/>'), analysis_style))
        story.append(Spacer(1, 0.3*inch))
        
        # License Plate Evidence
        if plate_images:
            story.append(Paragraph(f"📸 LICENSE PLATE EVIDENCE ({len(plate_images)} images detected)", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Sort by confidence (highest first)
            sorted_plates = sorted(plate_images, key=lambda x: x[1], reverse=True)
            
            # Add ALL plate images (no limit)
            for idx, (img_path, confidence) in enumerate(sorted_plates, 1):
                try:
                    # Add page break every 4 images to avoid overcrowding
                    if idx > 1 and (idx - 1) % 4 == 0:
                        story.append(PageBreak())
                        story.append(Paragraph(f"📸 LICENSE PLATE EVIDENCE (continued)", heading_style))
                        story.append(Spacer(1, 0.1*inch))
                    
                    img = Image(str(img_path), width=3*inch, height=1.5*inch)
                    story.append(img)
                    caption = Paragraph(
                        f"<b>Detection #{idx}</b> - Confidence: {confidence:.1%}",
                        caption_style
                    )
                    story.append(caption)
                    story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    print(f"Could not add image {img_path}: {e}")
        
        # Video Frame (if provided)
        if video_frame_path and Path(video_frame_path).exists():
            story.append(PageBreak())
            story.append(Paragraph("🎥 VIOLATION FRAME CAPTURE", heading_style))
            story.append(Spacer(1, 0.1*inch))
            try:
                frame_img = Image(str(video_frame_path), width=6*inch, height=4*inch)
                story.append(frame_img)
            except Exception as e:
                print(f"Could not add frame image: {e}")
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            f"Generated by Smart City Traffic Monitoring System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        return filename
    
    def generate_summary_report(
        self,
        violations: List[Dict[str, Any]],
        video_metadata: Dict[str, Any],
        ai_summary: str,
        ai_recommendations: str
    ) -> Path:
        """
        Generate an executive summary PDF for all violations
        """
        filename = self.output_dir / f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=26,
            textColor=colors.HexColor('#1976d2'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("📊 TRAFFIC VIOLATION EXECUTIVE SUMMARY", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Statistics
        stats_data = [
            ['Total Violations', str(len(violations))],
            ['Video Duration', video_metadata.get('duration', 'N/A')],
            ['Frames Processed', str(video_metadata.get('frames_processed', 0))],
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 0.4*inch))
        
        # AI Summary
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#d32f2f'),
            spaceAfter=12
        )
        story.append(Paragraph("🤖 AI ANALYSIS SUMMARY", heading_style))
        story.append(Paragraph(ai_summary.replace('\n', '<br/>'), styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        story.append(Paragraph("💡 RECOMMENDATIONS", heading_style))
        story.append(Paragraph(ai_recommendations.replace('\n', '<br/>'), styles['BodyText']))
        
        doc.build(story)
        return filename
