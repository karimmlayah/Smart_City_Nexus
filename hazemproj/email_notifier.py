"""
Email notification system for traffic violations
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ViolationEmailNotifier:
    """Send email notifications for traffic violations"""
    
    def __init__(
        self,
        smtp_host: str = "smtp.mail.yahoo.com",
        smtp_port: int = 587,
        email_user: str = "jbalihazem@yahoo.com",
        email_password: str = "pzrvuxystyxwtnjd"
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.email_user = email_user
        self.email_password = email_password
    
    def send_violation_email(
        self,
        to_email: str,
        owner_name: str,
        plate_number: str,
        violation_date: str,
        fine_amount: float = 85.0,
        plate_image_path: Optional[Path] = None
    ) -> bool:
        """
        Send violation notification email
        
        Args:
            to_email: Recipient email
            owner_name: Vehicle owner name
            plate_number: License plate number
            violation_date: Date/time of violation
            fine_amount: Fine amount in TND
            plate_image_path: Optional path to plate image
            
        Returns:
            True if email sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart('related')
            msg['From'] = self.email_user
            msg['To'] = to_email
            msg['Subject'] = f"⚠️ Avis de Contravention - Plaque {plate_number}"
            
            # HTML content
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #d32f2f 0%, #c62828 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header .icon {{
            font-size: 60px;
            margin-bottom: 10px;
        }}
        .content {{
            padding: 30px;
        }}
        .alert-box {{
            background-color: #ffebee;
            border-left: 4px solid #d32f2f;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .info-table td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .info-table td:first-child {{
            font-weight: bold;
            color: #555;
            width: 40%;
        }}
        .fine-amount {{
            background-color: #fff3e0;
            border: 2px solid #ff9800;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }}
        .fine-amount .amount {{
            font-size: 36px;
            font-weight: bold;
            color: #e65100;
        }}
        .fine-amount .currency {{
            font-size: 18px;
            color: #666;
        }}
        .plate-image {{
            text-align: center;
            margin: 20px 0;
        }}
        .plate-image img {{
            max-width: 100%;
            border: 3px solid #ddd;
            border-radius: 8px;
        }}
        .footer {{
            background-color: #f5f5f5;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #666;
        }}
        .button {{
            display: inline-block;
            padding: 12px 30px;
            background-color: #1976d2;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon">🚦</div>
            <h1>Avis de Contravention</h1>
            <p>Système de Surveillance Routière Intelligent</p>
        </div>
        
        <div class="content">
            <div class="alert-box">
                <strong>⚠️ ATTENTION:</strong> Une infraction au code de la route a été détectée pour votre véhicule.
            </div>
            
            <p>Bonjour <strong>{owner_name}</strong>,</p>
            
            <p>Nous vous informons qu'une infraction a été enregistrée pour le véhicule immatriculé <strong>{plate_number}</strong>.</p>
            
            <table class="info-table">
                <tr>
                    <td>📋 Type d'infraction:</td>
                    <td><strong>Franchissement de feu rouge</strong></td>
                </tr>
                <tr>
                    <td>📅 Date et heure:</td>
                    <td>{violation_date}</td>
                </tr>
                <tr>
                    <td>🚗 Plaque d'immatriculation:</td>
                    <td><strong>{plate_number}</strong></td>
                </tr>
                <tr>
                    <td>📍 Lieu:</td>
                    <td>Intersection surveillée - Tunis</td>
                </tr>
            </table>
            
            <div class="fine-amount">
                <div>Montant de l'amende</div>
                <div class="amount">{fine_amount:.2f} <span class="currency">TND</span></div>
            </div>
            
            <p><strong>Détails de l'infraction:</strong></p>
            <p>Votre véhicule a été détecté franchissant la ligne d'arrêt alors que le feu de signalisation était rouge. Cette infraction a été enregistrée par notre système de surveillance automatique équipé d'intelligence artificielle.</p>
            
            <p><strong>📸 Preuve photographique:</strong></p>
            <div class="plate-image">
                <img src="cid:plate_image" alt="Plaque d'immatriculation" style="max-width:400px;">
            </div>
            
            <p><strong>Modalités de paiement:</strong></p>
            <ul>
                <li>Paiement en ligne sur le portail des amendes</li>
                <li>Paiement dans les bureaux de poste</li>
                <li>Paiement dans les centres de perception</li>
            </ul>
            
            <p style="color:#d32f2f;"><strong>⏰ Délai de paiement: 30 jours</strong></p>
            <p style="font-size:12px;color:#666;">En cas de non-paiement dans les délais, l'amende sera majorée conformément à la réglementation en vigueur.</p>
            
            <p>Pour toute contestation, veuillez vous référer aux procédures légales en vigueur.</p>
        </div>
        
        <div class="footer">
            <p><strong>Système de Surveillance Routière Intelligent</strong></p>
            <p>Cet email a été généré automatiquement par notre système de détection des infractions.</p>
            <p>Pour plus d'informations: contact@traffic-monitoring.tn</p>
            <p style="margin-top:10px;color:#999;">© 2026 Smart City Traffic Monitoring System</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Attach HTML
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # Attach plate image if provided
            if plate_image_path and plate_image_path.exists():
                with open(plate_image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-ID', '<plate_image>')
                    msg.attach(image)
            
            # Send email
            print(f"\n📧 Sending violation email...")
            print(f"   To: {to_email}")
            print(f"   Plate: {plate_number}")
            print(f"   Fine: {fine_amount} TND")
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            print(f"   ✓ Email sent successfully!")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed to send email: {e}")
            import traceback
            traceback.print_exc()
            return False
