import os
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "DB-Client-Id": os.getenv("DB_CLIENT_ID"),
    "DB-Api-Key":   os.getenv("DB_API_KEY"),
    "Accept":       "application/xml",
}

BASE = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"

STATIONS = {
    "Vaihingen (Enz)":        "8006053",
    "Sersheim":               "8005540",
    "Sachsenheim":            "8005253",
    "Bietigheim-Bissingen":   "8000038",
    "Asperg":                 "8000630",
    "Stuttgart-Zuffenhausen": "8005778",
    "Stuttgart-Feuerbach":    "8005770",
    "Stuttgart Hbf":          "8000096",
    "Ludwigsburg":            "8000235",
    "Stuttgart Stadtmitte":   "8006700",
    "Stuttgart Feuersee":     "8006699",
    "Stuttgart Universität":  "8006513",
}


def parse_time(t):
    if not t:
        return None
    return datetime.strptime(t, "%y%m%d%H%M")


def fetch_plan(eva: str) -> ET.Element:
    now = datetime.now()
    r = requests.get(
        f"{BASE}/plan/{eva}/{now.strftime('%y%m%d')}/{now.strftime('%H')}",
        headers=HEADERS, timeout=15,
    )
    r.raise_for_status()
    return ET.fromstring(r.text)


def fetch_changes(eva: str) -> ET.Element:
    r = requests.get(
        f"{BASE}/fchg/{eva}",
        headers=HEADERS, timeout=15,
    )
    r.raise_for_status()
    return ET.fromstring(r.text)


def build_changes_map(fchg_root: ET.Element) -> dict:
    return {s.get("id"): s for s in fchg_root.findall("s")}
