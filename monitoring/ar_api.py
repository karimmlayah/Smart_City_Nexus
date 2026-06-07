"""Dynamic mock API payloads for MedinaMind AR page — Tunis live data."""
import hashlib
import random
from zoneinfo import ZoneInfo

from django.utils import timezone

TUNIS_TZ = ZoneInfo("Africa/Tunis")
CITY_NAME = "Tunis"
COUNTRY = "Tunisia"
REGION = "Greater Tunis"

MODULE_DEFS = [
    {
        "id": "traffic",
        "name": "Traffic",
        "icon": "🚗",
        "status": "active",
        "description": "Live traffic flow monitoring in Tunis",
    },
    {
        "id": "energy",
        "name": "Energy",
        "icon": "⚡",
        "status": "stable",
        "description": "Tunis city energy usage monitoring",
    },
    {
        "id": "pollution",
        "name": "Pollution",
        "icon": "☁️",
        "status": "good",
        "description": "Air quality tracking across Tunis",
    },
    {
        "id": "security",
        "name": "Security",
        "icon": "🛡️",
        "status": "safe",
        "description": "Urban safety monitoring in Tunis",
    },
]

MODULE_LABELS = {m["id"]: m["name"] for m in MODULE_DEFS}

# Typical daily temperature band for Tunis by month (min night, max afternoon)
TUNIS_TEMP_RANGES = {
    1: (11, 16), 2: (11, 17), 3: (13, 20), 4: (15, 23),
    5: (18, 27), 6: (22, 32), 7: (25, 35), 8: (25, 36),
    9: (22, 31), 10: (18, 27), 11: (14, 21), 12: (12, 17),
}


def tunis_now():
    return timezone.now().astimezone(TUNIS_TZ)


def _stable_rng(bucket_key: str, salt: str = "") -> random.Random:
    digest = hashlib.md5(f"{bucket_key}:{salt}".encode(), usedforsecurity=False).hexdigest()
    return random.Random(int(digest[:8], 16))


def _tunis_temp(month: int, hour: int) -> int:
    """Deterministic temperature from month + hour (realistic daily curve)."""
    lo, hi = TUNIS_TEMP_RANGES.get(month, (20, 28))
    span = hi - lo

    if hour <= 5:
        temp = lo
    elif hour <= 9:
        temp = lo + span * 0.22 * (hour - 5) / 4
    elif hour <= 15:
        temp = lo + span * (0.22 + 0.73 * (hour - 9) / 6)
    elif hour <= 20:
        temp = hi - span * 0.38 * (hour - 15) / 5
    else:
        temp = lo + span * 0.18

    return round(temp)


def _condition_for_temp(temp: int) -> str:
    if temp >= 32:
        return "Hot"
    if temp >= 28:
        return "Sunny"
    if temp >= 24:
        return "Partly Cloudy"
    if temp >= 18:
        return "Clear"
    if temp >= 14:
        return "Mild"
    return "Cool"


def _humidity_for_temp(temp: int, month: int) -> int:
    if month in (6, 7, 8):
        humidity = 58 - int((temp - 22) * 1.1)
        return max(38, min(58, humidity))
    if month in (12, 1, 2):
        humidity = 62 + int((18 - temp) * 0.8)
        return max(55, min(72, humidity))
    humidity = 55 - int((temp - 20) * 0.6)
    return max(45, min(68, humidity))


def _wind_for_hour(hour: int) -> int:
    """Stable wind speed — changes only every 3 hours."""
    slot = hour // 3
    speeds = (10, 12, 16, 20, 18, 14, 11, 9)
    return speeds[slot % len(speeds)]


def _feels_like(temp: int, humidity: int) -> int:
    if humidity >= 55 and temp >= 26:
        return temp + 1
    if humidity <= 42 and temp >= 28:
        return temp
    return temp + (1 if temp >= 27 else 0)


def _tunis_weather(now) -> dict:
    """Weather derived from Tunis local time — stable within each hour."""
    month = now.month
    hour = now.hour
    temp = _tunis_temp(month, hour)
    humidity = _humidity_for_temp(temp, month)
    wind = _wind_for_hour(hour)
    feels = _feels_like(temp, humidity)

    return {
        "temperature": f"{temp}°C",
        "temperature_c": temp,
        "condition": _condition_for_temp(temp),
        "humidity": f"{humidity}%",
        "wind": f"{wind} km/h",
        "feels_like": f"{feels}°C",
    }


def build_ar_live_stats():
    now = tunis_now()
    # Values refresh every 15 minutes — no wild jumps every few seconds
    bucket = now.strftime("%Y-%m-%d-%H") + f"-{now.minute // 15}"
    rng = _stable_rng(bucket, "stats")

    energy = rng.randint(96, 108)
    safety = rng.randint(94, 97)
    parking = rng.randint(1580, 2140)
    traffic_flow = rng.randint(58, 82)
    traffic_status = rng.choice(["Smooth", "Moderate", "Busy"])
    air_quality = rng.choice(["Good", "Moderate", "Excellent"])

    return {
        "success": True,
        "city": CITY_NAME,
        "country": COUNTRY,
        "region": REGION,
        "timezone": "Africa/Tunis (UTC+1)",
        "timestamp": now.isoformat(),
        "last_updated": now.strftime("%H:%M:%S"),
        "population": "2.8M",
        "energy_usage": f"{energy} MW",
        "air_quality": air_quality,
        "safety": f"{safety}%",
        "parking_spots": parking,
        "traffic_status": traffic_status,
        "traffic_flow": f"{traffic_flow}%",
        "traffic": f"{traffic_status} · {traffic_flow}%",
    }


def build_ar_live_info():
    now = tunis_now()
    weather = _tunis_weather(now)

    return {
        "success": True,
        "city": f"{CITY_NAME}, {COUNTRY}",
        "region": REGION,
        "timezone": "Africa/Tunis (UTC+1)",
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%d/%m/%Y"),
        "weekday": now.strftime("%A"),
        "weather": weather,
    }


def build_ar_modules():
    now = tunis_now()
    bucket = now.strftime("%Y-%m-%d")
    rng = _stable_rng(bucket, "modules")

    modules = []
    for mod in MODULE_DEFS:
        entry = dict(mod)
        entry["status"] = rng.choice(["active", "stable", "good", "safe"])
        modules.append(entry)

    return {
        "success": True,
        "city": CITY_NAME,
        "modules": modules,
    }
