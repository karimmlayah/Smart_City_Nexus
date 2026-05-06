"""
Database module for vehicle registration and owner information
"""
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "vehicles.db"


class VehicleDatabase:
    """Manage vehicle registration database"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with tables and sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create vehicles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE NOT NULL,
                plate_left TEXT NOT NULL,
                plate_right TEXT NOT NULL,
                owner_name TEXT NOT NULL,
                phone_number TEXT,
                email TEXT NOT NULL,
                address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create violations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER,
                plate_number TEXT NOT NULL,
                violation_type TEXT DEFAULT 'RED_LIGHT',
                violation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fine_amount REAL DEFAULT 85.0,
                video_token TEXT,
                car_tracking_id INTEGER,
                email_sent BOOLEAN DEFAULT 0,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✓ Database initialized: {self.db_path}")
    
    def add_sample_data(self):
        """Add sample vehicle data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sample vehicles - using static email for testing
        sample_vehicles = [
            ("85 TN 207", "85", "207", "Ahmed Ben Ali", "+216 98 123 456", "hazem.jbali@esprit.tn", "Tunis, Tunisia"),
            ("6835 TN 210", "6835", "210", "Fatma Trabelsi", "+216 22 345 678", "hazem.jbali@esprit.tn", "Sfax, Tunisia"),
            ("123 TN 4567", "123", "4567", "Mohamed Gharbi", "+216 55 789 012", "hazem.jbali@esprit.tn", "Sousse, Tunisia"),
            ("200 TN 9271", "200", "9271", "Leila Mansour", "+216 24 567 890", "hazem.jbali@esprit.tn", "Ariana, Tunisia"),
        ]
        
        for plate, left, right, name, phone, email, address in sample_vehicles:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO vehicles 
                    (plate_number, plate_left, plate_right, owner_name, phone_number, email, address)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (plate, left, right, name, phone, email, address))
            except sqlite3.IntegrityError:
                pass  # Already exists
        
        conn.commit()
        conn.close()
        print(f"✓ Sample data added")
    
    def normalize_plate(self, plate_text: str) -> tuple:
        """
        Normalize plate number for database search
        Handles: "85 تونس 207", "85 TN 207", "85-207", "85_207", "85.207", "85 207", "85207", etc.
        Returns: (normalized_plate, left_numbers, right_numbers)
        """
        # Replace Arabic with TN
        plate_text = plate_text.replace('تونس', 'TN')
        
        # Replace common separators with TN
        separators = ['-', '_', '.', '/', '|', ':', ';', ',', '*', '+', '=']
        for sep in separators:
            if sep in plate_text:
                plate_text = plate_text.replace(sep, ' TN ')
                break
        
        # Clean up spaces
        plate_text = ' '.join(plate_text.split())
        
        # Extract all numbers
        numbers = re.findall(r'\d+', plate_text)
        
        if len(numbers) >= 2:
            # Multiple number groups found
            left = numbers[0]
            right = numbers[-1]
            normalized = f"{left} TN {right}"
            return normalized, left, right
        elif len(numbers) == 1:
            # Single number group - might be concatenated like "85207"
            num = numbers[0]
            if len(num) >= 3:
                # Try to split intelligently
                # Tunisian plates typically: 1-4 digits + 1-4 digits
                # Try different split points
                best_split = None
                
                # Try splitting at different positions
                for split_pos in range(1, len(num)):
                    left_part = num[:split_pos]
                    right_part = num[split_pos:]
                    
                    # Prefer splits where both parts are 1-4 digits
                    if 1 <= len(left_part) <= 4 and 1 <= len(right_part) <= 4:
                        best_split = (left_part, right_part)
                        break
                
                if best_split:
                    left, right = best_split
                    normalized = f"{left} TN {right}"
                    return normalized, left, right
                else:
                    # Fallback: split in middle
                    mid = len(num) // 2
                    left = num[:mid]
                    right = num[mid:]
                    normalized = f"{left} TN {right}"
                    return normalized, left, right
            else:
                # Too short, can't split
                return f"{num} TN ?", num, ""
        
        # No numbers found - return as is
        return plate_text, "", ""
    
    def search_vehicle(self, plate_text: str) -> Optional[Dict[str, Any]]:
        """
        Search for vehicle by plate number
        Tries multiple matching strategies
        """
        normalized, left, right = self.normalize_plate(plate_text)
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Strategy 1: Exact match
        cursor.execute("SELECT * FROM vehicles WHERE plate_number = ?", (normalized,))
        result = cursor.fetchone()
        
        if not result and left and right:
            # Strategy 2: Match by left and right numbers
            cursor.execute("""
                SELECT * FROM vehicles 
                WHERE plate_left = ? AND plate_right = ?
            """, (left, right))
            result = cursor.fetchone()
        
        if not result and left and right:
            # Strategy 3: Try flipped (OCR might reverse left/right)
            cursor.execute("""
                SELECT * FROM vehicles 
                WHERE plate_left = ? AND plate_right = ?
            """, (right, left))
            result = cursor.fetchone()
            
            if result:
                print(f"   ⚠ Found match with FLIPPED numbers: {right} TN {left}")
        
        conn.close()
        
        if result:
            return dict(result)
        return None
    
    def record_violation(
        self,
        plate_number: str,
        vehicle_id: Optional[int],
        video_token: str,
        car_tracking_id: int,
        fine_amount: float = 85.0
    ) -> int:
        """Record a violation in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO violations 
            (vehicle_id, plate_number, violation_type, fine_amount, video_token, car_tracking_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (vehicle_id, plate_number, "RED_LIGHT", fine_amount, video_token, car_tracking_id))
        
        violation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return violation_id
    
    def mark_email_sent(self, violation_id: int):
        """Mark that email has been sent for this violation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE violations SET email_sent = 1 WHERE id = ?
        """, (violation_id,))
        
        conn.commit()
        conn.close()
    
    def get_all_vehicles(self) -> List[Dict[str, Any]]:
        """Get all registered vehicles"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM vehicles ORDER BY id")
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def get_violations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent violations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT v.*, veh.owner_name, veh.email 
            FROM violations v
            LEFT JOIN vehicles veh ON v.vehicle_id = veh.id
            ORDER BY v.violation_date DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]


# Initialize database on import
db = VehicleDatabase()
db.add_sample_data()
