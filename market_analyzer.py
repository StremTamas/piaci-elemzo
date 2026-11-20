import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from email.message import EmailMessage
import json
from pathlib import Path
import smtplib
import ssl
import time

from google import genai
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import streamlit as st
import yfinance as yf

# Ikon elérési útja (a market_analyzer.py mellett a repó gyökerében)
ICON_PATH = Path(__file__).parent / "ikon.png"

# Oldal beállítása
st.set_page_config(
    page_title="AI Piaci Iránytű",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🧭",
    layout="wide",
)

# --- EGYEDI DIZÁJN: NARANCS SZÍNVILÁG & MODERN SZÖGLETES BETŰKÉSZLET ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, .exo-title {
        font-family: 'Exo 2', sans-serif !important;
        letter-spacing: 0.02em;
    }

    .main-banner {
        background: linear-gradient(135deg, #1E1E24 0%, #2B2D38 100%);
        border-left: 6px solid #FF6B00;
        padding: 18px 22px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #FFFFFF;
    }
    .main-banner h1 {
        margin: 0;
        font-size: 1.65rem;
        color: #FFFFFF;
    }
    .main-banner p {
        margin: 4px 0 0 0;
        color: #CBD5E1;
        font-size: 0.92rem;
    }

    .fx-strip {
        background: rgba(255, 107, 0, 0.10);
        border: 1px solid rgba(255, 107, 0, 0.35);
        border-left: 5px solid #FF6B00;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 14px;
        font-size: 0.88rem;
        font-weight: 600;
        line-height: 1.45;
    }

    .asset-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        border-bottom: 2px solid rgba(255, 107, 0, 0.25);
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    .asset-name {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.22rem;
        font-weight: 800;
        color: #FF6B00;
    }
    .asset-ticker {
        font-family: 'Exo 2', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        opacity: 0.75;
        margin-left: 6px;
    }

    .decision-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 10px 0 12px 0;
    }
    .decision-box {
        padding: 10px 12px;
        border-radius: 6px;
        border-left: 4px solid #FF6B00;
        background: rgba(255, 107, 0, 0.07);
    }
    .decision-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.8;
        font-weight: 700;
        margin-bottom: 3px;
    }
    .decision-value {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.02rem;
        font-weight: 800;
    }

    .badge-buy {
        border-left-color: #10B981 !important;
        background: rgba(16, 185, 129, 0.12) !important;
        color: #059669;
    }
    .badge-hold {
        border-left-color: #FF6B00 !important;
        background: rgba(255, 107, 0, 0.12) !important;
        color: #EA580C;
    }
    .badge-sell {
        border-left-color: #EF4444 !important;
        background: rgba(239, 68, 68, 0.12) !important;
        color: #DC2626;
    }

    @media (prefers-color-scheme: dark) {
        .badge-buy { color: #34D399; }
        .badge-hold { color: #FB923C; }
        .badge-sell { color: #F87171; }
    }

    .tech-pill {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        background: rgba(255, 107, 0, 0.12);
        border: 1px solid rgba(255, 107, 0, 0.35);
        margin-bottom: 10px;
    }

    .legend-bar {
        font-size: 0.78rem;
        opacity: 0.85;
        margin: -6px 0 10px 0;
    }

    div.stButton > button {
        background-color: #FF6B00 !important;
        color: #FFFFFF !important;
        font-family: 'Exo 2', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #E05A00 !important;
        box-shadow: 0 4px 12px rgba(255, 107, 0, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_icon_base64():
    """Base64 formátumba alakítja a logó ikont, ha megtalálható a repóban."""
    if ICON_PATH.exists():
        return base64.b64encode(ICON_PATH.read_bytes()).decode("utf-8")
    return None


def get_clean_gemini_key():
    """Kinyeri és megtisztítja a Gemini API kulcsot a felesleges szóközöktől/idézőjelektől."""
    raw_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or ""
    return str(raw_key).strip().strip('"').strip("'")


GEMINI_API_KEY = get_clean_gemini_key()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _send_gemini_request(client, model_name, prompt):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text or ""


def call_gemini_single_batch(prompt):
    """Kötegelt kérést küld az ingyenes Gemini API-nak, 503-as terhelési csúcs esetén rövid újrapróbálkozással."""
    if not GEMINI_API_KEY:
        raise RuntimeError("Hiányzik a GEMINI_API_KEY a Streamlit Secrets beállításokból.")

    client = genai.Client(vertexai=False, api_key=GEMINI_API_KEY)
    last_err = None

    for model_name in ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]:
        for attempt in range(3):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_send_gemini_request, client, model_name, prompt)
            try:
                result_text = future.result(timeout=15)
                executor.shutdown(wait=False)
                return result_text
            except FuturesTimeoutError:
                executor.shutdown(wait=False)
                last_err = TimeoutError(f"A modell ({model_name}) 15 másodpercen belül nem válaszolt.")
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            except Exception as e:
                executor.shutdown(wait=False)
                last_err = e
                err_upper = str(e).upper()
                if "503" in err_upper or "UNAVAILABLE" in err_upper:
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))  # 2 mp, majd 4 mp várakozás
                        continue
                    break
                if "429" in err_upper or "RESOURCE_EXHAUSTED" in err_upper:
                    break
                if any(code in err_upper for code in ["API_KEY", "API KEY", "400", "403", "404", "NOT_FOUND"]):
                    break
                raise e

    if last_err is not None:
        raise last_err
    raise RuntimeError("Nem sikerült kapcsolódni a Gemini API-hoz.")


def get_gspread_client():
    """Hitelesített Google Sheets kliens létrehozása."""
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_resource
def get_alert_history():
    """Szerverszintű memória a már elküldött riasztások nyilvántartására."""
    return {}


def send_gmail_alert(subject, plain_body, html_body):
    """E-mail küldése Gmail SMTP-n keresztül."""
    sender_email = st.secrets["SMTP_EMAIL"]
    app_password = st.secrets["SMTP_APP_PASSWORD"]
    recipient_email = st.secrets.get("ALERT_RECIPIENT_EMAIL", sender_email)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"AI Piaci Iránytű <{sender_email}>"
    msg["To"] = recipient_email
    msg.set_content(plain_body)
    msg.add_alternative(html_body, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)


def check_and_send_alert(item):
    """Riasztást küld, ha KITÁRAZÁS vagy HÍGÍTÁS / ÉRDEMES VENNI jelzés alakul ki."""
    existing_act = item.get("existing_action", "").upper()
    new_act = item.get("new_entry_action", "").upper()
    tech = item.get("tech") or {}
    rsi = tech.get("rsi_14")

    triggers = []
    if "KITÁRAZÁS" in existing_act:
        triggers.append(f"Portfólió jelzés: {item['existing_action']}")
    if "ÉRDEMES VENNI" in new_act or "HÍGÍTÁS" in existing_act:
        rsi_txt = f" (RSI: {rsi:.1f})" if rsi is not None else ""
        triggers.append(f"Kedvező beszállási / hígítási zóna{rsi_txt}")

    if not triggers:
        return False, None

    alert_key = f"{item['name']}_{item['ticker']}_{'_'.join(triggers)}"
    history = get_alert_history()
    now = datetime.now()

    if alert_key in history and (now - history[alert_key]) < timedelta(hours=24):
        return False, "Már riasztva (24h)"

    trigger_text = " | ".join(triggers)
    price_str = f"{tech['price']:,.4f}" if tech.get("price") is not None else "N/A"
    change_str = f"{tech['change_pct']:+.2f}%" if tech.get("change_pct") is not None else "N/A"

    subject = f"🔥 Piaci Jelzés: {item['name']} ({item['ticker']}) – {item['existing_action']}"

    plain_body = (
        f"Piaci jelzés érkezett:\n\n"
        f"Eszköz: {item['name']} ({item['ticker']})\n"
        f"Kiváltó ok: {trigger_text}\n"
        f"Aktuális ár: {price_str} ({change_str})\n"
        f"Saját pozícióra: {item['existing_action']}\n"
        f"Új vásárlásra: {item['new_entry_action']}\n\n"
        f"Indoklás:\n{item['reasoning']}\n"
    )

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #222; line-height: 1.5;">
        <div style="border-left: 5px solid #FF6B00; padding-left: 14px;">
          <h2 style="color: #FF6B00; margin-bottom: 4px;">🔥 {item['name']} ({item['ticker']})</h2>
          <p style="margin-top: 0;"><strong>Jelzés:</strong> {trigger_text}</p>
        </div>
        <table style="border-collapse: collapse; width: 100%; max-width: 520px; margin: 12px 0;">
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; background: #f8f9fa;"><strong>Aktuális ár</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{price_str} ({change_str})</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; background: #f8f9fa;"><strong>Meglévő pozícióra</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>{item['existing_action']}</strong></td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; background: #f8f9fa;"><strong>Új vásárlásra</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>{item['new_entry_action']}</strong></td>
          </tr>
        </table>
        <p><strong>Elemzés:</strong><br>{item['reasoning']}</p>
      </body>
    </html>
    """

    try:
        send_gmail_alert(subject, plain_body, html_body)
        history[alert_key] = now
        return True, "Riasztás elküldve"
    except Exception as e:
        return False, f"SMTP hiba: {e}"


def _fetch_single_fx_pair(ticker_symbol, pair_label):
    """Lekéri egy konkrét devizapár (pl. EUR/HUF vagy USD/HUF) 1 éves pozícióját."""
    try:
        hist = yf.Ticker(ticker_symbol).history(period="1y", interval="1d", timeout=10)
        if hist is None or hist.empty or "Close" not in hist:
            return None

        close = hist["Close"].dropna()
        if len(close) < 2:
            return None

        rate = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        chg = ((rate - prev) / prev) * 100
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else rate
        low_1y = float(close.min())
        high_1y = float(close.max())
        pos_pct = ((rate - low_1y) / (high_1y - low_1y) * 100) if high_1y > low_1y else 50.0

        if pos_pct >= 75:
            regime_short = "Gyenge HUF zóna (éves csúcs közelében – profitrealizálásra kedvező)"
            zone_tag = "WEAK_HUF"
        elif pos_pct <= 25:
            regime_short = "Erős HUF zóna (éves mélypont közelében – devizaváltásra kedvező)"
            zone_tag = "STRONG_HUF"
        else:
            regime_short = "Középmezőnyben mozgó forintárfolyam"
            zone_tag = "NEUTRAL_HUF"

        return {
            "pair": pair_label,
            "rate": rate,
            "change_pct": chg,
            "sma50": sma50,
            "pos_pct": pos_pct,
            "zone_tag": zone_tag,
            "regime_short": regime_short,
            "summary": f"{pair_label}: {rate:.2f} Ft ({chg:+.2f}%) | 50n átlag: {sma50:.2f} Ft | {regime_short}",
        }
    except Exception:
        return None


def fetch_fx_context():
    """Külön-külön lekéri az EUR/HUF és az USD/HUF devizakörnyezetet."""
    return {
        "EUR": _fetch_single_fx_pair("EURHUF=X", "EUR/HUF"),
        "USD": _fetch_single_fx_pair("USDHUF=X", "USD/HUF"),
    }


def get_active_assets_from_sheet(client):
    """Beolvassa az 'IGEN' státuszú eszközöket az 'Eszkozok' munkalapról."""
    sheet = client.open_by_key(st.secrets["SHEET_ID"]).worksheet("Eszkozok")
    records = sheet.get_all_records()

    return [
        row for row in records
        if str(row.get("aktiv", "")).strip().upper() == "IGEN" and row.get("ticker")
    ]


def get_vig_funds_from_sheet(client):
    """Beolvassa a 'VIG_Alapok' munkalapról a rögzített VIG befektetési alapok árfolyamsorát."""
    try:
        spreadsheet = client.open_by_key(st.secrets["SHEET_ID"])
        vig_sheet = spreadsheet.worksheet("VIG_Alapok")
    except Exception:
        return []

    records = vig_sheet.get_all_records()
    grouped = {}

    for row in records:
        if str(row.get("aktiv", "IGEN")).strip().upper() != "IGEN":
            continue
        fund_name = str(row.get("alap_nev", "")).strip()
        date_str = str(row.get("datum", "")).strip()
        raw_price = str(row.get("arfolyam", "")).replace(" ", "").replace(",", ".").strip()
        currency = str(row.get("deviza", "HUF")).strip().upper() or "HUF"

        if not fund_name or not raw_price:
            continue

        try:
            price_val = float(raw_price)
        except ValueError:
            continue

        grouped.setdefault(fund_name, {"currency": currency, "points": []})
        grouped[fund_name]["points"].append((date_str, price_val))

    funds = []
    for fund_name, info in grouped.items():
        pts = sorted(info["points"], key=lambda x: x[0])
        dates = pd.to_datetime([p[0] for p in pts], errors="coerce")
        prices = [p[1] for p in pts]
        idx = dates if not dates.isna().all() else range(len(prices))
        df = pd.DataFrame({"Close": prices}, index=idx)
        funds.append({
            "name": fund_name,
            "ticker": f"VIG ({info['currency']})",
            "currency": info["currency"],
            "hist_df": df,
        })

    return funds


def log_results_to_sheet(client, results, run_timestamp):
    """Kötegelve menti az elemzési eredményeket a Google Táblázat 'Naplo' munkalapjára."""
    if not results:
        return False, "Nincs menthető adat."

    spreadsheet = client.open_by_key(st.secrets["SHEET_ID"])
    headers = [
        "Időpont",
        "Eszköz",
        "Ticker",
        "Árfolyam",
        "Napi változás (%)",
        "RSI (14)",
        "SMA 50",
        "SMA 200",
        "AI Pontszám",
        "Ajánlás",
        "Technikai kép",
        "Kockázat",
        "AI Indoklás",
    ]

    try:
        naplo_sheet = spreadsheet.worksheet("Naplo")
    except gspread.WorksheetNotFound:
        naplo_sheet = spreadsheet.add_worksheet(title="Naplo", rows=1000, cols=len(headers))
        naplo_sheet.append_row(headers, value_input_option="USER_ENTERED")

    if len(naplo_sheet.get_all_values()) == 0:
        naplo_sheet.append_row(headers, value_input_option="USER_ENTERED")

    rows_to_append = []
    for item in results:
        tech = item.get("tech") or {}
        price_val = round(tech["price"], 4) if tech.get("price") is not None else ""
        change_val = round(tech["change_pct"], 2) if tech.get("change_pct") is not None else ""
        rsi_val = round(tech["rsi_14"], 2) if tech.get("rsi_14") is not None else ""
        sma50_val = round(tech["sma_50"], 4) if tech.get("sma_50") is not None else ""
        sma200_val = round(tech["sma_200"], 4) if tech.get("sma_200") is not None else ""

        rows_to_append.append([
            run_timestamp,
            item["name"],
            item["ticker"],
            price_val,
            change_val,
            rsi_val,
            sma50_val,
            sma200_val,
            item["score"],
            f"Portfólió: {item['existing_action']}",
            f"Új vétel: {item['new_entry_action']}",
            item["plain_tech"],
            item["reasoning"],
        ])

    naplo_sheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
    return True, f"Naplózva a 'Naplo' munkalapra ({len(rows_to_append)} sor)."


def calculate_technical_indicators(hist_df):
    """Kiszámolja a 14 napos RSI-t, az 50/200 napos mozgóátlagot és előkészíti a grafikont."""
    if hist_df is None or hist_df.empty or "Close" not in hist_df:
        return None

    close = hist_df["Close"].dropna()
    if len(close) == 0:
        return None

    current_price = float(close.iloc[-1])
    if len(close) >= 2 and float(close.iloc[-2]) != 0:
        prev_close = float(close.iloc[-2])
        change_pct = ((current_price - prev_close) / prev_close) * 100
    else:
        change_pct = 0.0

    sma_50_series = close.rolling(window=50).mean()
    sma_200_series = close.rolling(window=200).mean()

    sma_50 = float(sma_50_series.iloc[-1]) if len(close) >= 50 and pd.notna(sma_50_series.iloc[-1]) else None
    sma_200 = float(sma_200_series.iloc[-1]) if len(close) >= 200 and pd.notna(sma_200_series.iloc[-1]) else None

    rsi_14 = None
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        if pd.notna(rsi_series.iloc[-1]):
            rsi_14 = float(rsi_series.iloc[-1])

    if sma_50 and sma_200:
        if current_price > sma_50 > sma_200:
            trend = "Erős emelkedő trend (SMA 50 és SMA 200 felett)"
        elif current_price < sma_50 < sma_200:
            trend = "Tartós csökkenő trend (mindkét mozgóátlag alatt)"
        elif current_price < sma_200:
            trend = "Megtört főtrend (az SMA 200 alá esett)"
        elif sma_50 > sma_200:
            trend = "Emelkedő főtrend, átmeneti visszahúzódással"
        else:
            trend = "Csökkenő főtrend, rövid távú felpattanással"
    elif len(close) >= 2:
        first_val = float(close.iloc[0])
        total_chg = ((current_price - first_val) / first_val) * 100 if first_val else 0.0
        trend = f"Rögzített időszak elmozdulása: {total_chg:+.2f}%"
    else:
        trend = "Egyetlen rögzített árfolyamadat áll rendelkezésre"

    chart_df = pd.DataFrame({
        "Árfolyam": close,
        "SMA 200": sma_200_series,
        "SMA 50": sma_50_series,
    }).tail(252)

    if getattr(chart_df.index, "tz", None) is not None:
        chart_df.index = chart_df.index.tz_localize(None)

    return {
        "price": current_price,
        "change_pct": change_pct,
        "rsi_14": rsi_14,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "trend": trend,
        "chart_df": chart_df,
    }


def evaluate_technical_matrix(tech, currency="USD", fx_map=None):
    """
    EGYSÉGES DÖNTÉSI MÁTRIX (Trend × RSI × Saját Devizapár):
    Minden eszközt a saját devizaneméhez tartozó forintárfolyam (EUR/HUF vagy USD/HUF) alapján finomhangol.
    """
    fx_map = fx_map or {}
    relevant_fx = fx_map.get(currency) if currency in ("EUR", "USD") else None
    zone_tag = relevant_fx.get("zone_tag") if relevant_fx else None
    weak_huf = zone_tag == "WEAK_HUF"
    strong_huf = zone_tag == "STRONG_HUF"

    if relevant_fx:
        fx_note = (
            f"💱 **Devizahatás ({relevant_fx['pair']} = {relevant_fx['rate']:.2f} Ft):** "
            f"{relevant_fx['regime_short']} (1 éves sáv {relevant_fx['pos_pct']:.0f}%-ánál)."
        )
    elif currency == "HUF":
        fx_note = "💱 **Devizahatás (HUF):** Forintos eszköz, devizaváltási árfolyamhatás nem érinti."
    else:
        fx_note = f"💱 **Devizahatás ({currency}):** Nincs külön devizapár-korrekció."

    if not tech:
        return {
            "existing_action": "TARTÁS",
            "new_entry_action": "KIVÁRÁS",
            "score": 50,
            "plain_summary": "Nincs elegendő historikus árfolyamadat.",
            "reasoning": "Nincs elegendő historikus árfolyamadat az értékeléshez.",
            "trend_note": "Nincs elegendő adat a trend megállapításához.",
            "rsi_note": "Nincs elegendő adat az RSI kiszámításához.",
            "fx_note": fx_note,
            "matrix_rule_note": "Adathiány miatt alapértelmezett kivárás.",
            "rsi_badge": "Nincs adat",
        }

    price = tech.get("price")
    sma50 = tech.get("sma_50")
    sma200 = tech.get("sma_200")
    rsi = tech.get("rsi_14")

    # 1. Hosszú/középtávú trend besorolása (SMA 50 & SMA 200)
    if price is not None and sma50 is not None and sma200 is not None:
        if price > sma50 and sma50 > sma200:
            trend_state = "UP_STRONG"
            trend_note = "✅ **Hosszú/középtávú trend (SMA 50 & 200):** Az árfolyam mindkét mozgóátlag felett van (`Ár > SMA 50 > SMA 200`), az emelkedő főtrend ép."
        elif price >= sma200:
            trend_state = "UP_PULLBACK"
            trend_note = "⚖️ **Hosszú/középtávú trend (SMA 50 & 200):** Az árfolyam az SMA 50 alá korrigált, de az SMA 200 még tartja (`SMA 200 ≤ Ár ≤ SMA 50`)."
        elif sma50 < sma200:
            trend_state = "DOWN_CONFIRMED"
            trend_note = "🔴 **Hosszú/középtávú trend (SMA 50 & 200):** Az árfolyam mindkét mozgóátlag alatt van, és az SMA 50 is az SMA 200 alá bukott (`Ár < SMA 50 < SMA 200`)."
        else:
            trend_state = "DOWN_BROKEN"
            trend_note = "🟠 **Hosszú/középtávú trend (SMA 50 & 200):** Az árfolyam letörte a 200 napos átlagot (`Ár < SMA 200`), a hosszú távú emelkedő trend megtört."
    elif price is not None and sma50 is not None:
        if price >= sma50:
            trend_state = "UP_STRONG"
            trend_note = "✅ **Középtávú trend (SMA 50):** Az árfolyam az 50 napos átlag felett mozog."
        else:
            trend_state = "DOWN_BROKEN"
            trend_note = "🟠 **Középtávú trend (SMA 50):** Az árfolyam az 50 napos átlag alá esett."
    else:
        trend_state = "NEUTRAL"
        trend_note = f"ℹ️ **Trend:** {tech.get('trend', 'Kevés historikus adat')}."

    # 2. Rövid távú lendület besorolása (RSI 14)
    if rsi is None:
        rsi_state = "UNKNOWN"
        rsi_badge = "Nincs adat"
        rsi_note = "ℹ️ **Rövid távú lendület (RSI 14):** Kevés adat a számításhoz."
    elif rsi < 30:
        rsi_state = "OVERSOLD_DEEP"
        rsi_badge = "Mélyen túladott (<30)"
        rsi_note = f"🟢 **Rövid távú lendület (RSI = {rsi:.1f}):** 30 alatti mélyen túladott zóna – rövid távon kimerülőben az eladói nyomás, statisztikailag gyakori a technikai felpattanás."
    elif rsi < 35:
        rsi_state = "OVERSOLD_NEAR"
        rsi_badge = "Túladott közeli (30–35)"
        rsi_note = f"🟢 **Rövid távú lendület (RSI = {rsi:.1f}):** 30–35 közötti túladott-közeli zóna – már olcsó tartomány, érdemes figyelni a fordulós gyertyákat."
    elif rsi > 70:
        rsi_state = "OVERBOUGHT"
        rsi_badge = "Túlvett (>70)"
        rsi_note = f"🟠 **Rövid távú lendület (RSI = {rsi:.1f}):** 70 feletti túlvett zóna – rövid távon túlhevült az emelkedés, nő a visszahúzódás esélye."
    elif rsi > 65:
        rsi_state = "ELEVATED"
        rsi_badge = "Emelkedett (65–70)"
        rsi_note = f"🟡 **Rövid távú lendület (RSI = {rsi:.1f}):** 65–70 közötti magasabb tartomány – erős lendület, de új beszállásra már nem olcsó."
    else:
        rsi_state = "NEUTRAL"
        rsi_badge = "Semleges (35–65)"
        rsi_note = f"🔵 **Rövid távú lendület (RSI = {rsi:.1f}):** 35 és 65 közötti kiegyensúlyozott tartomány."

    # 3. Kombinált mátrix döntés (Trend × RSI × Saját Devizapár)
    if trend_state in ("UP_STRONG", "UP_PULLBACK"):
        if rsi_state in ("OVERSOLD_DEEP", "OVERSOLD_NEAR"):
            existing_act = "HÍGÍTÁS (DCA)"
            new_act = "ÉRDEMES VENNI"
            score = 84 if strong_huf else 78
            if strong_huf and relevant_fx:
                fx_extra = f" Ráadásul az {relevant_fx['pair']} ({relevant_fx['rate']:.2f} Ft) erős forint zónában jár, így a devizaváltás is kedvező."
            elif weak_huf and relevant_fx:
                fx_extra = f" Megjegyzés: az {relevant_fx['pair']} ({relevant_fx['rate']:.2f} Ft) gyenge forint zónában van, így forintból váltva drágább a deviza."
            else:
                fx_extra = ""
            reason = (
                f"Emelkedő főtrendben (SMA 200 felett) rövid távon túladott zónába ({rsi:.1f}) esett az RSI: "
                f"kedvező beszállási és hígítási lehetőség.{fx_extra}"
            )
            matrix_note = "🎯 **Kombinált döntés (Emelkedő trend + Alacsony RSI):** Meglévő pozícióra **HÍGÍTÁS (DCA)**, új vételre **ÉRDEMES VENNI**."
        elif rsi_state == "OVERBOUGHT":
            if weak_huf and relevant_fx:
                existing_act = "RÉSZLEGES KITÁRAZÁS"
                reason = (
                    f"Túlvett RSI ({rsi:.1f}) mellett az {relevant_fx['pair']} ({relevant_fx['rate']:.2f} Ft) gyenge forint zónában (éves csúcs közelében) jár: "
                    "kedvező alkalom részleges profitrealizálásra forintra, új vétellel kivárni."
                )
            else:
                existing_act = "TARTÁS"
                reason = (
                    f"Az emelkedő trend ép, de a 14 napos RSI ({rsi:.1f}) már túlvett zónában van: "
                    "meglévő pozíciót érdemes tartani, új vétellel megvárni egy visszahúzódást."
                )
            new_act = "KIVÁRÁS"
            score = 62
            matrix_note = f"🎯 **Kombinált döntés (Emelkedő trend + Túlvett RSI):** Meglévő pozícióra **{existing_act}**, új vételre **KIVÁRÁS**."
        else:
            existing_act = "TARTÁS"
            new_act = "KIVÁRÁS"
            score = 68 if trend_state == "UP_STRONG" else 58
            if strong_huf and relevant_fx:
                fx_extra = f" (Az erős forint miatt {relevant_fx['pair']} devizát előre váltani kedvező, de a részvényvétellel érdemes támaszra várni.)"
            else:
                fx_extra = ""
            reason = (
                "Az emelkedő főtrend ép és az RSI semleges tartományban mozog: "
                f"meglévő pozíciót érdemes tartani, új vásárlással pedig egy támasz tesztelésére várni.{fx_extra}"
            )
            matrix_note = "🎯 **Kombinált döntés (Emelkedő trend + Semleges RSI):** Meglévő pozícióra **TARTÁS**, új vételre **KIVÁRÁS**."

    elif trend_state in ("DOWN_BROKEN", "DOWN_CONFIRMED"):
        if rsi_state in ("OVERSOLD_DEEP", "OVERSOLD_NEAR"):
            existing_act = "TARTÁS"
            new_act = "KIVÁRÁS"
            score = 42
            reason = (
                f"Bár a rövid távú RSI ({rsi:.1f}) már túladott (eladni a pánik alján már késő, jöhet felpattanás), "
                "az árfolyam az SMA 200 alatt van: új vétellel a hulló kés miatt ki kell várni az RSI 35 fölé fordulását."
            )
            matrix_note = "🎯 **Kombinált döntés (SMA 200 alatt + Alacsony RSI = Hulló kés):** Pánikból eladni már késő (**Meglévő: TARTÁS**), de új pénzzel meg kell várni az RSI 35 fölé fordulását (**Új vétel: KIVÁRÁS**)."
        else:
            if trend_state == "DOWN_CONFIRMED":
                existing_act = "RÉSZLEGES KITÁRAZÁS"
                new_act = "KERÜLENDŐ"
                score = 28
                reason = (
                    "Tartós csökkenő trend (SMA 50 és SMA 200 alatt) mellett az RSI már nem túladott: "
                    "az új vétel kerülendő, meglévő pozíciónál kockázatcsökkentés mérlegelendő."
                )
                matrix_note = "🎯 **Kombinált döntés (Tartós eső trend + Nem túladott RSI):** Meglévő pozícióra **RÉSZLEGES KITÁRAZÁS**, új vétel **KERÜLENDŐ**."
            else:
                existing_act = "TARTÁS"
                new_act = "KIVÁRÁS"
                score = 40
                reason = (
                    "Az árfolyam az SMA 200 alá esett és iránykeresés zajlik: "
                    "új vásárlással ki kell várni a stabil támasz kialakulását."
                )
                matrix_note = "🎯 **Kombinált döntés (SMA 200 alatt + Semleges RSI):** Meglévő pozícióra **TARTÁS**, új vételre **KIVÁRÁS**."
    else:
        existing_act = "TARTÁS"
        new_act = "KIVÁRÁS"
        score = 50
        reason = tech.get("trend", "Vegyes technikai kép, kivárás javasolt.")
        matrix_note = "🎯 **Kombinált döntés:** Meglévő pozícióra **TARTÁS**, új vételre **KIVÁRÁS**."

    return {
        "existing_action": existing_act,
        "new_entry_action": new_act,
        "score": score,
        "plain_summary": tech.get("trend", ""),
        "reasoning": reason,
        "trend_note": trend_note,
        "rsi_note": rsi_note,
        "fx_note": fx_note,
        "matrix_rule_note": matrix_note,
        "rsi_badge": rsi_badge,
    }


def search_news_fallback(query_text):
    """Név alapú hírkeresés a yfinance beépített keresőjével."""
    items = []
    if not hasattr(yf, "Search"):
        return items
    try:
        search_res = yf.Search(query_text, news_count=3)
        raw = getattr(search_res, "news", []) or []
        for idx, item in enumerate(raw[:3], start=1):
            title = item.get("title", "")
            publisher = item.get("publisher", "Forrás")
            url = item.get("link", "")
            if title:
                items.append({
                    "id": idx,
                    "title": title,
                    "publisher": publisher,
                    "url": url,
                })
    except Exception:
        pass
    return items


def fetch_yahoo_data(ticker_symbol, asset_name):
    """Lekéri a 2 éves grafikonadatokat és a legfontosabb híreket (időtúllépés-védelemmel)."""
    ticker = yf.Ticker(ticker_symbol)

    try:
        hist = ticker.history(period="2y", interval="1d", timeout=10)
        tech_data = calculate_technical_indicators(hist)
    except Exception:
        tech_data = None

    news_items = []
    try:
        raw_news = ticker.news[:3] if ticker.news else []
        for idx, item in enumerate(raw_news, start=1):
            content = item.get("content", item)
            title = content.get("title", "")
            provider = content.get("provider", {})
            publisher = provider.get("displayName") if isinstance(provider, dict) else item.get("publisher", "Forrás")
            url = (
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
                or item.get("link")
                or ""
            )
            if title:
                news_items.append({
                    "id": idx,
                    "title": title,
                    "publisher": publisher,
                    "url": url,
                })
    except Exception:
        pass

    if not news_items and asset_name:
        news_items = search_news_fallback(asset_name)

    return tech_data, news_items


def analyze_all_assets_batch(raw_items, fx_map):
    """EGYETLEN kötegelt Gemini API hívással elemzi az összes eszközt és fordítja le a híreket."""
    eur_summary = fx_map["EUR"]["summary"] if fx_map.get("EUR") else "EUR/HUF adat nem elérhető."
    usd_summary = fx_map["USD"]["summary"] if fx_map.get("USD") else "USD/HUF adat nem elérhető."

    assets_payload = []
    for item in raw_items:
        tech = item["tech"]
        if tech:
            rsi_str = f"{tech['rsi_14']:.1f}" if tech["rsi_14"] is not None else "N/A"
            sma50_str = f"{tech['sma_50']:,.4f}" if tech["sma_50"] is not None else "N/A"
            sma200_str = f"{tech['sma_200']:,.4f}" if tech["sma_200"] is not None else "N/A"
            tech_txt = (
                f"Ár: {tech['price']:,.4f} ({tech['change_pct']:+.2f}%), "
                f"RSI14: {rsi_str}, SMA50: {sma50_str}, SMA200: {sma200_str}, Trend: {tech['trend']}"
            )
        else:
            tech_txt = "Nincs technikai adat."

        assets_payload.append({
            "id": item["id"],
            "name": item["name"],
            "ticker": item["ticker"],
            "currency": item["currency"],
            "technical": tech_txt,
            "headlines": [n["title"] for n in item["raw_news"]],
        })

    prompt = f"""
Gyakorlatias, kockázattudatos befektetési tanácsadóként viselkedsz egy magyar befektető számára.
Elemezd az alábbi összes portfólióeszközt EGYETLEN válaszban.

--- DEVIZAKÖRNYEZET (SAJÁT DEVIZÁHOZ IGAZÍTVA) ---
- EUR eszközökhöz: {eur_summary}
- USD eszközökhöz: {usd_summary}
- HUF eszközökhöz: devizahatás nem értelmezhető, kizárólag a saját trendre és hírekre fókuszálj.

--- DÖNTÉSI SZABÁLYOK (SZIGORÚAN TARTSD BE) ---
1. Emelkedő trendben (Ár > SMA 200) és alacsony RSI (< 35) mellett: Meglévő = "HÍGÍTÁS (DCA)", Új vétel = "ÉRDEMES VENNI". (Ha az adott eszköz devizájában erős a forint, említsd meg a kedvező devizaváltási lehetőséget is.)
2. Megtört főtrendben (Ár < SMA 200) és alacsony RSI (< 32) mellett ("hulló kés"): pánikból eladni már késő (Meglévő = "TARTÁS"), de új vételhez meg kell várni az RSI 35 fölé fordulását és a hírek tisztázódását (Új vétel = "KIVÁRÁS"). Erős forint önmagában NEM indokol vételt!
3. Tartós csökkenő trendben (Ár < SMA 50 < SMA 200) és nem túladott RSI (>= 32) mellett: Meglévő = "RÉSZLEGES KITÁRAZÁS", Új vétel = "KERÜLENDŐ".
4. Túlvett RSI (> 70) esetén: ha az adott eszköz saját devizája (EUR vagy USD) gyenge forint zónában van, javasolj "RÉSZLEGES KITÁRAZÁS"-t forintra, különben "TARTÁS"-t; új vételre mindenképp "KIVÁRÁS"-t.

--- ESZKÖZÖK LISTÁJA ---
{json.dumps(assets_payload, ensure_ascii=False)}

Feladat minden egyes "id" kulcshoz:
1. Dönts külön a MEGLÉVŐ pozícióról és az ÚJ vásárlásról a technikai adatok + hírek + az eszköz saját devizapárja alapján.
2. Fordítsd le magyarra a kapott angol/idegen nyelvű hírcímeket ("headlines").

Válaszolj KIZÁRÓLAG érvényes JSON objektummal, ahol a legfelső szintű kulcsok az eszközök "id" értékei ("A0", "A1", ...), és minden eszköz alatt pontosan ezek a kulcsok szerepelnek:
- "existing_portfolio_action": pontosan egy ezekből: "TARTÁS", "HÍGÍTÁS (DCA)", "RÉSZLEGES KITÁRAZÁS", "TELJES KITÁRAZÁS"
- "new_entry_action": pontosan egy ezekből: "ÉRDEMES VENNI", "KIVÁRÁS", "KERÜLENDŐ"
- "plain_tech_summary": 1 rövid magyar mondat szakszavak (RSI/SMA) nélkül arról, hol áll az árfolyam
- "sentiment_score": egész szám 1 és 100 között
- "reasoning": 2 tömör magyar mondat arról, miért ezt javaslod most (sződd bele a híreket és az eszköz saját devizahatását is)
- "translated_news": lista a lefordított hírcímekkel (ha nem volt hír, [])
"""

    try:
        raw_text = call_gemini_single_batch(prompt)
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Az AI válasz nem tartalmazott érvényes JSON objektumot.")
        parsed_map = json.loads(raw_text[start_idx:end_idx + 1])
        return parsed_map, None
    except Exception as e:
        return {}, str(e)


def get_badge_class(action_text):
    """Meghatározza a kártyadoboz színét a döntés alapján."""
    txt = action_text.upper()
    if "ÉRDEMES VENNI" in txt or "HÍGÍTÁS" in txt:
        return "badge-buy"
    if "KITÁRAZÁS" in txt or "KERÜLENDŐ" in txt:
        return "badge-sell"
    return "badge-hold"


def infer_currency_from_ticker(ticker):
    """Meghatározza az eszköz várható devizanemét a tickerből."""
    t = ticker.upper()
    if t.endswith(".BD"):
        return "HUF"
    if t.endswith(".DE") or t.endswith(".PA") or t.endswith(".AS") or "-EUR" in t:
        return "EUR"
    if t.endswith("=X"):
        return "FX"
    return "USD"


def format_price(val):
    """Egységes árfolyamformázás."""
    if val is None:
        return "N/A"
    return f"{val:,.4f}" if val < 10 else f"{val:,.2f}"


def render_technical_learning_panel(tech, currency="USD", fx_map=None):
    """Látható SMA 50 / SMA 200 / RSI értékelő és az egységes mátrix szerinti tanuló magyarázat."""
    if not tech:
        return

    price = tech.get("price")
    sma50 = tech.get("sma_50")
    sma200 = tech.get("sma_200")
    rsi = tech.get("rsi_14")
    eval_res = evaluate_technical_matrix(tech, currency, fx_map)

    st.markdown("##### 📐 Technikai mutatók és döntési logika (Tanuló nézet)")

    t1, t2, t3 = st.columns(3)

    if sma50 is not None and price is not None:
        diff50 = ((price - sma50) / sma50) * 100
        t1.metric("SMA 50 (középtáv)", format_price(sma50), f"{diff50:+.1f}% az árhoz képest")
    else:
        t1.metric("SMA 50 (középtáv)", "N/A", "Kevés adat (<50 nap)")

    if sma200 is not None and price is not None:
        diff200 = ((price - sma200) / sma200) * 100
        t2.metric("SMA 200 (hosszú táv)", format_price(sma200), f"{diff200:+.1f}% az árhoz képest")
    else:
        t2.metric("SMA 200 (hosszú táv)", "N/A", "Kevés adat (<200 nap)")

    if rsi is not None:
        t3.metric("RSI (14 napos)", f"{rsi:.1f}", eval_res["rsi_badge"])
    else:
        t3.metric("RSI (14 napos)", "N/A", "Kevés adat (<15 nap)")

    st.info(
        f"{eval_res['trend_note']}\n\n"
        f"{eval_res['rsi_note']}\n\n"
        f"{eval_res['fx_note']}\n\n"
        f"{eval_res['matrix_rule_note']}"
    )

    with st.expander("📘 Döntési mátrix puska: Mikor vételi jel az RSI < 30, és hogyan számít a deviza?"):
        st.markdown(
            """
- **Emelkedő főtrend (`Ár > SMA 200`) + Alacsony RSI (`< 35`):** Egészséges trendben átmeneti visszaesés → **HÍGÍTÁS (DCA) / ÉRDEMES VENNI**. Ha közben az adott eszköz devizájához képest erős a forint, dupla előny (olcsó részvény + olcsó deviza).
- **Megtört főtrend (`Ár < SMA 200`) + Alacsony RSI (`< 32`) – „Hulló kés”:** Eladni a pánik alján már késő (**Meglévő: TARTÁS**), de új pénzzel meg kell várni, amíg az RSI visszafordul **35 fölé**, és tisztulnak a hírek (**Új vétel: KIVÁRÁS**). Az erős forint önmagában **nem** indokol részvényvételt.
- **Tartós eső trend (`Ár < SMA 50 < SMA 200`) + Nem túladott RSI (`≥ 32`):** A felpattanások kockázatcsökkentésre valók → **RÉSZLEGES KITÁRAZÁS / Új vétel: KERÜLENDŐ**.
- **Túlvett piac (`RSI > 70`):** Rövid távon túlhevült → **Új vétel: KIVÁRÁS**, devizás eszköznél gyenge forint mellett (az adott EUR/HUF vagy USD/HUF sáv tetején) **RÉSZLEGES KITÁRAZÁS**.
            """
        )


def render_asset_details(item, fx_map=None):
    """Megjeleníti egy eszköz grafikonját, a tanuló SMA/RSI értékelőt és a magyar híreket."""
    tech = item["tech"]
    if tech and tech.get("chart_df") is not None and len(tech["chart_df"]) >= 2:
        st.line_chart(
            tech["chart_df"],
            color=["#FF6B00", "#0EA5E9", "#64748B"],
            height=210,
        )
        st.markdown(
            '<div class="legend-bar">📈 <strong>Grafikon:</strong> '
            '<span style="color:#FF6B00;font-weight:700;">● Árfolyam</span> &nbsp;|&nbsp; '
            '<span style="color:#64748B;font-weight:700;">● SMA 50 (50 napos átlag)</span> &nbsp;|&nbsp; '
            '<span style="color:#0EA5E9;font-weight:700;">● SMA 200 (200 napos átlag)</span></div>',
            unsafe_allow_html=True,
        )

    render_technical_learning_panel(tech, item.get("currency", "USD"), fx_map)

    with st.expander("📰 Elemzett hírek (magyarra fordítva)"):
        if item["news_list"]:
            for n in item["news_list"]:
                if n["url"]:
                    st.markdown(f"- [{n['hu_title']}]({n['url']}) *({n['publisher']})*")
                else:
                    st.markdown(f"- {n['hu_title']} *({n['publisher']})*")
        else:
            st.write("Nincsenek friss hírek az elmúlt napokból.")


def run_full_analysis_with_progress():
    """Élő folyamatjelzővel futtatja az adatgyűjtést és az 1-hívásos kötegelt AI elemzést."""
    status_box = st.empty()
    progress_bar = st.progress(0)

    status_box.info("1/4 • Csatlakozás a Google Táblázathoz...")
    client = get_gspread_client()
    assets = get_active_assets_from_sheet(client)
    vig_funds = get_vig_funds_from_sheet(client)
    progress_bar.progress(15)

    status_box.info("2/4 • EUR/HUF és USD/HUF devizakörnyezet lekérése...")
    fx_map = fetch_fx_context()
    progress_bar.progress(25)

    raw_items = []
    total_count = len(assets) + len(vig_funds)

    for idx, asset in enumerate(assets):
        ticker = str(asset["ticker"]).strip()
        name = str(asset.get("name", ticker)).strip()
        currency = infer_currency_from_ticker(ticker)

        if total_count > 0:
            pct = 25 + int((idx / total_count) * 45)
            progress_bar.progress(min(70, pct))
            status_box.info(f"3/4 • Árfolyamok letöltése ({idx + 1}/{total_count}): **{name}** ({ticker})...")

        tech_data, news_items = fetch_yahoo_data(ticker, name)
        raw_items.append({
            "id": f"A{len(raw_items)}",
            "name": name,
            "ticker": ticker,
            "currency": currency,
            "tech": tech_data,
            "raw_news": news_items,
        })

    if vig_funds:
        vig_news = search_news_fallback("VIG Asset Management")
        for f_idx, fund in enumerate(vig_funds):
            step_num = len(assets) + f_idx + 1
            if total_count > 0:
                pct = 25 + int(((step_num - 1) / total_count) * 45)
                progress_bar.progress(min(70, pct))
                status_box.info(f"3/4 • VIG alap feldolgozása ({step_num}/{total_count}): **{fund['name']}**...")

            tech_data = calculate_technical_indicators(fund["hist_df"])
            raw_items.append({
                "id": f"A{len(raw_items)}",
                "name": fund["name"],
                "ticker": fund["ticker"],
                "currency": fund["currency"],
                "tech": tech_data,
                "raw_news": vig_news,
            })

    if not raw_items:
        progress_bar.empty()
        status_box.empty()
        return [], fx_map, datetime.now().strftime("%Y.%m.%d. %H:%M"), False, "Nincs aktív eszköz.", None

    progress_bar.progress(78)
    status_box.info(f"4/4 • Összesített AI elemzés és hírfordítás ({len(raw_items)} eszköz 1 csomagban)...")
    ai_map, ai_err = analyze_all_assets_batch(raw_items, fx_map)

    results = []
    for item in raw_items:
        ai_res = ai_map.get(item["id"])
        if not ai_res:
            eval_res = evaluate_technical_matrix(item["tech"], item["currency"], fx_map)
            prefix = "[Szabályalapú mód] " if ai_err else ""
            ai_res = {
                "existing_portfolio_action": eval_res["existing_action"],
                "new_entry_action": eval_res["new_entry_action"],
                "plain_tech_summary": eval_res["plain_summary"],
                "sentiment_score": eval_res["score"],
                "reasoning": f"{prefix}{eval_res['reasoning']}",
                "translated_news": [],
            }

        translated_titles = ai_res.get("translated_news", [])
        paired_news = []
        for n_idx, orig in enumerate(item["raw_news"]):
            hu_title = translated_titles[n_idx] if n_idx < len(translated_titles) else orig["title"]
            paired_news.append({
                "hu_title": hu_title,
                "publisher": orig["publisher"],
                "url": orig["url"],
            })

        results.append({
            "name": item["name"],
            "ticker": item["ticker"],
            "currency": item["currency"],
            "tech": item["tech"],
            "existing_action": ai_res.get("existing_portfolio_action", "TARTÁS"),
            "new_entry_action": ai_res.get("new_entry_action", "KIVÁRÁS"),
            "plain_tech": ai_res.get("plain_tech_summary", ""),
            "score": ai_res.get("sentiment_score", 50),
            "reasoning": ai_res.get("reasoning", ""),
            "news_list": paired_news,
        })

    progress_bar.progress(92)
    run_timestamp = datetime.now().strftime("%Y.%m.%d. %H:%M")

    try:
        log_ok, log_msg = log_results_to_sheet(client, results, run_timestamp)
    except Exception as e:
        log_ok, log_msg = False, f"Naplózási hiba: {e}"

    progress_bar.progress(100)
    progress_bar.empty()
    status_box.empty()

    return results, fx_map, run_timestamp, log_ok, log_msg, ai_err


@st.cache_data(ttl=3600, show_spinner=False)
def get_shared_analysis():
    """1 órán át szerverszinten tárolja az eredményt, így sem a frissítés, sem a megosztás nem fogyaszt új kvótát."""
    return run_full_analysis_with_progress()


# --- FELÜLET (UI) ---
st.markdown(
    """
    <div class="main-banner">
        <h1 style="display: flex; align-items: center; gap: 12px;">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0;">
                <path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z" fill="#FF6B00"/>
                <path d="M19 3L19.8 5.2L22 6L19.8 6.8L19 9L18.2 6.8L16 6L18.2 5.2L19 3Z" fill="#FF6B00" opacity="0.75"/>
            </svg>
            <span>AI PIACI IRÁNYTŰ</span>
        </h1>
        <p>Gyors portfólió-döntéstámogató • Magyar blue chipek, nemzetközi eszközök, VIG alapok & EUR/USD/HUF hatás</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    icon_b64 = get_icon_base64()
    if icon_b64:
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 10px 12px;
                margin-bottom: 14px;
                background: linear-gradient(135deg, #1E1E24 0%, #2B2D38 100%);
                border: 1px solid rgba(255, 107, 0, 0.35);
                border-left: 4px solid #FF6B00;
                border-radius: 6px;
            ">
                <img src="data:image/png;base64,{icon_b64}" style="
                    width: 48px;
                    height: 48px;
                    border-radius: 6px;
                    border: 1px solid rgba(255, 107, 0, 0.5);
                    object-fit: cover;
                    flex-shrink: 0;
                " />
                <div>
                    <div style="font-family: 'Exo 2', sans-serif; font-weight: 800; font-size: 0.96rem; color: #FFFFFF; letter-spacing: 0.03em; line-height: 1.2;">
                        AI PIACI IRÁNYTŰ
                    </div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.72rem; color: #CBD5E1; margin-top: 2px;">
                        Portfólió &amp; Deviza Radar
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ⚡ Vezérlőpult")
    if st.button("🔄 Frissítés most (Élő adat)", use_container_width=True):
        get_shared_analysis.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📧 E-mail Riasztások")
    alerts_enabled = st.toggle("Automatikus riasztás", value=True)
    st.caption("Jelzést küld **KITÁRAZÁS**, **HÍGÍTÁS** vagy **ÉRDEMES VENNI** jelzés esetén.")

    if st.button("Teszt e-mail küldése", use_container_width=True):
        try:
            send_gmail_alert(
                "🔥 Teszt riasztás – AI Piaci Iránytű",
                "A Gmail SMTP kapcsolat megfelelően működik.",
                "<p>A <strong>Gmail SMTP</strong> kapcsolat megfelelően működik.</p>",
            )
            st.success("Teszt e-mail sikeresen elküldve!")
        except Exception as e:
            st.error(f"Hiba a küldéskor: {e}")

try:
    data, fx_map, last_updated, log_ok, log_msg, ai_err = get_shared_analysis()
except Exception as e:
    st.error(f"Hiba az adatok betöltésekor: {e}")
    st.stop()

if ai_err:
    get_shared_analysis.clear()
    err_u = ai_err.upper()
    if "429" in err_u or "RESOURCE_EXHAUSTED" in err_u:
        st.info(
            "ℹ️ **AI napi/perces kvótalimit elérve:** A portfólió automatikusan az egységesített "
            "SMA 50 / SMA 200 / RSI + EUR/USD devizamátrix alapján értékelve jelenik meg."
        )
    elif "503" in err_u or "UNAVAILABLE" in err_u or "TIMEOUT" in err_u:
        st.info(
            "ℹ️ **Az AI szerver átmenetileg túlterhelt:** A portfólió automatikusan az egységesített "
            "SMA 50 / SMA 200 / RSI + EUR/USD devizamátrix alapján értékelve jelenik meg."
        )
    else:
        st.warning(f"⚠️ **AI kiegészítő megjegyzés** (szabályalapú mód aktív): `{ai_err}`")

if fx_map and (fx_map.get("EUR") or fx_map.get("USD")):
    fx_lines = []
    if fx_map.get("EUR"):
        fx_lines.append(f"🇪🇺 {fx_map['EUR']['summary']}")
    if fx_map.get("USD"):
        fx_lines.append(f"🇺🇸 {fx_map['USD']['summary']}")
    st.markdown(f'<div class="fx-strip">{"<br>".join(fx_lines)}</div>', unsafe_allow_html=True)

st.caption(f"⏱️ Frissítve: **{last_updated}** &nbsp;|&nbsp; 💾 {log_msg if log_ok else '⚠️ ' + log_msg}")

if not data:
    st.warning("Nincs 'IGEN' státuszú eszköz az 'Eszkozok' vagy 'VIG_Alapok' munkalapon.")
    st.stop()

alert_map = {}
if alerts_enabled:
    for item in data:
        item_uid = f"{item['name']}_{item['ticker']}"
        sent, status_msg = check_and_send_alert(item)
        if status_msg:
            alert_map[item_uid] = (sent, status_msg)
            if sent:
                st.toast(f"Riasztás elküldve: {item['name']}", icon="🔥")

ctrl_col1, ctrl_col2 = st.columns([2, 2])
with ctrl_col1:
    view_mode = st.radio(
        "Megjelenítés",
        ["📋 Listás nézet", "🗂️ Kártyás nézet"],
        horizontal=True,
        label_visibility="collapsed",
    )
with ctrl_col2:
    filter_option = st.selectbox(
        "Szűrés jelzésre",
        [
            "Minden eszköz",
            "🟢 Csak vétel / hígítás",
            "🔴 Csak kitárazás / kerülendő",
        ],
        label_visibility="collapsed",
    )

filtered_data = []
for item in data:
    ex_u = item["existing_action"].upper()
    new_u = item["new_entry_action"].upper()
    if filter_option == "🟢 Csak vétel / hígítás":
        if "ÉRDEMES VENNI" in new_u or "HÍGÍTÁS" in ex_u:
            filtered_data.append(item)
    elif filter_option == "🔴 Csak kitárazás / kerülendő":
        if "KITÁRAZÁS" in ex_u or "KERÜLENDŐ" in new_u:
            filtered_data.append(item)
    else:
        filtered_data.append(item)

if not filtered_data:
    st.info("A kiválasztott szűrőnek jelenleg egyetlen eszköz sem felel meg.")
    st.stop()

# --- 1. LISTÁS NÉZET ---
if view_mode == "📋 Listás nézet":
    table_rows = []
    for item in filtered_data:
        tech = item["tech"]
        price_str = format_price(tech["price"]) if tech else "N/A"
        chg_str = f"{tech['change_pct']:+.2f}%" if tech else "N/A"

        table_rows.append({
            "Eszköz": f"{item['name']} ({item['ticker']})",
            "Deviza": item["currency"],
            "Árfolyam": price_str,
            "Napi %": chg_str,
            "Saját pozíció": item["existing_action"],
            "Új vásárlás": item["new_entry_action"],
            "Helyzetkép": item["plain_tech"],
            "Indoklás": item["reasoning"],
        })

    df_list = pd.DataFrame(table_rows)
    table_height = min(680, 42 + len(df_list) * 38)
    st.dataframe(df_list, use_container_width=True, hide_index=True, height=table_height)

    st.markdown("---")
    selected_label = st.selectbox(
        "🔎 Grafikon, SMA/RSI értékelő és magyarított hírek megtekintése ehhez az eszközhöz:",
        [f"{item['name']} ({item['ticker']})" for item in filtered_data],
    )
    selected_item = next(
        (x for x in filtered_data if f"{x['name']} ({x['ticker']})" == selected_label),
        None,
    )
    if selected_item:
        with st.container(border=True):
            st.markdown(
                f"**{selected_item['name']} ({selected_item['ticker']})** — "
                f"Saját pozíció: **{selected_item['existing_action']}** | "
                f"Új vétel: **{selected_item['new_entry_action']}**"
            )
            st.write(selected_item["reasoning"])
            render_asset_details(selected_item, fx_map)

# --- 2. KÁRTYÁS NÉZET ---
else:
    cols = st.columns(2)
    for idx, item in enumerate(filtered_data):
        item_uid = f"{item['name']}_{item['ticker']}"
        alert_status = alert_map.get(item_uid)
        with cols[idx % 2]:
            with st.container(border=True):
                tech = item["tech"]

                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(
                        f'<div class="asset-header"><div><span class="asset-name">{item["name"]}</span>'
                        f'<span class="asset-ticker">{item["ticker"]}</span></div></div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    if tech:
                        st.metric("Árfolyam", format_price(tech["price"]), f"{tech['change_pct']:+.2f}%")
                    else:
                        st.metric("Árfolyam", "N/A")

                ex_class = get_badge_class(item["existing_action"])
                new_class = get_badge_class(item["new_entry_action"])

                st.markdown(
                    f"""
                    <div class="decision-grid">
                        <div class="decision-box {ex_class}">
                            <div class="decision-label">Saját pozícióra</div>
                            <div class="decision-value">{item['existing_action']}</div>
                        </div>
                        <div class="decision-box {new_class}">
                            <div class="decision-label">Új vásárlásra</div>
                            <div class="decision-value">{item['new_entry_action']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if item["plain_tech"]:
                    st.markdown(f'<div class="tech-pill">🧭 {item["plain_tech"]}</div>', unsafe_allow_html=True)

                st.write(item["reasoning"])

                if alert_status:
                    _, msg_text = alert_status
                    st.caption(f"🔔 Riasztás: **{msg_text}**")

                render_asset_details(item, fx_map)
