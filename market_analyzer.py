import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf
from google import genai
import pandas as pd
import json
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timedelta

# Oldal beállítása
st.set_page_config(page_title="AI Piaci Iránytű", page_icon="🔥", layout="wide")

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
        font-size: 0.9rem;
        font-weight: 600;
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


def get_clean_gemini_key():
    """Kinyeri és megtisztítja a Gemini API kulcsot a felesleges szóközöktől/idézőjelektől."""
    raw_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or ""
    return str(raw_key).strip().strip('"').strip("'")


GEMINI_API_KEY = get_clean_gemini_key()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def call_gemini_single_batch(prompt):
    """Egyetlen kötegelt kérést küld a Gemini API-nak, 503-as szervercsúcs esetén rövid újrapróbálkozással."""
    if not GEMINI_API_KEY:
        raise RuntimeError("Hiányzik a GEMINI_API_KEY a Streamlit Secrets beállításokból.")

    primary_vertex = GEMINI_API_KEY.startswith("AQ.")
    modes_to_try = [primary_vertex, not primary_vertex]
    last_err = None

    for use_vertex in modes_to_try:
        for attempt in range(3):
            try:
                if use_vertex:
                    client = genai.Client(vertexai=True, api_key=GEMINI_API_KEY)
                else:
                    client = genai.Client(api_key=GEMINI_API_KEY)

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                )
                return response.text or ""
            except Exception as e:
                last_err = e
                err_upper = str(e).upper()
                if "503" in err_upper or "UNAVAILABLE" in err_upper:
                    if attempt < 2:
                        time.sleep(3)
                        continue
                    raise e
                if "429" in err_upper or "RESOURCE_EXHAUSTED" in err_upper:
                    raise e
                if any(code in err_upper for code in ["API_KEY", "API KEY", "400", "403"]):
                    break
                raise e

    raise last_err

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
    """Riasztást küld, ha KITÁRAZÁS javasolt vagy az RSI 30 alatti."""
    existing_act = item.get("existing_action", "").upper()
    tech = item.get("tech") or {}
    rsi = tech.get("rsi_14")

    triggers = []
    if "KITÁRAZÁS" in existing_act:
        triggers.append(f"Portfólió jelzés: {item['existing_action']}")
    if rsi is not None and rsi < 30:
        triggers.append(f"Túladott zóna – Hígítási/beszállási esély (RSI: {rsi:.1f})")

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


def fetch_eurhuf_context():
    """Lekéri az EUR/HUF árfolyamhelyzetet a devizás profitrealizálás időzítéséhez."""
    try:
        hist = yf.Ticker("EURHUF=X").history(period="1y", interval="1d", timeout=10)
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
            regime = "Gyenge forint zóna (éves csúcs közelében – devizás profitrealizálásra kedvező)"
        elif pos_pct <= 25:
            regime = "Erős forint zóna (éves mélypont közelében – devizás vételre kedvező)"
        else:
            regime = "Középmezőnyben mozgó forintárfolyam"

        return {
            "rate": rate,
            "change_pct": chg,
            "sma50": sma50,
            "pos_pct": pos_pct,
            "summary": f"EUR/HUF: {rate:.2f} Ft ({chg:+.2f}%) | 50 napos átlag: {sma50:.2f} Ft | {regime}",
        }
    except Exception:
        return None


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
            trend = "Erős emelkedő trendben mozog"
        elif current_price < sma_50 < sma_200:
            trend = "Tartós csökkenő trendben mozog"
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


def generate_rule_based_fallback(tech_data, currency, fx_context, reason_prefix=""):
    """Szabályalapú döntés és magyar összefoglaló, ha az AI kvóta átmenetileg kimerült."""
    if not tech_data:
        return {
            "existing_portfolio_action": "TARTÁS",
            "new_entry_action": "KIVÁRÁS",
            "plain_tech_summary": "Árfolyamadatok feldolgozás alatt.",
            "sentiment_score": 50,
            "reasoning": f"{reason_prefix}Nincs elegendő historikus árfolyamadat az értékeléshez.",
            "translated_news": [],
        }

    price = tech_data.get("price")
    sma50 = tech_data.get("sma_50")
    sma200 = tech_data.get("sma_200")
    rsi = tech_data.get("rsi_14")

    existing_act = "TARTÁS"
    new_act = "KIVÁRÁS"
    score = 55
    reasons = []

    if price and sma50 and sma200:
        if price > sma50 > sma200:
            existing_act = "TARTÁS"
            score = 72
            reasons.append("Az árfolyam az 50 és 200 napos mozgóátlag felett halad, az emelkedő trend ép.")
        elif price < sma50 < sma200:
            existing_act = "RÉSZLEGES KITÁRAZÁS"
            new_act = "KERÜLENDŐ"
            score = 30
            reasons.append("Az árfolyam mindkét fő mozgóátlag alá esett, csökkenő trend alakult ki.")
        else:
            reasons.append("Az árfolyam a két fő mozgóátlag között konszolidálódik.")
    else:
        reasons.append(tech_data.get("trend", "Vegyes technikai kép."))

    if rsi is not None:
        if rsi < 30:
            existing_act = "HÍGÍTÁS (DCA)"
            new_act = "ÉRDEMES VENNI"
            score = 78
            reasons.append(f"A 14 napos RSI ({rsi:.1f}) túladott zónába esett, kedvező beszállási szint.")
        elif rsi > 72:
            new_act = "KIVÁRÁS"
            if currency in ("EUR", "USD") and fx_context and fx_context.get("pos_pct", 50) >= 75:
                existing_act = "RÉSZLEGES KITÁRAZÁS"
                reasons.append(f"Magas RSI ({rsi:.1f}) és gyenge forint mellett kedvező a részleges profitrealizálás.")
            else:
                reasons.append(f"A 14 napos RSI ({rsi:.1f}) túlvett tartományban van, érdemes kivárni a korrekciót.")

    return {
        "existing_portfolio_action": existing_act,
        "new_entry_action": new_act,
        "plain_tech_summary": tech_data.get("trend", ""),
        "sentiment_score": score,
        "reasoning": f"{reason_prefix}{' '.join(reasons)}",
        "translated_news": [],
    }


def analyze_all_assets_batch(raw_items, fx_context):
    """EGYETLEN kötegelt Gemini API hívással elemzi az összes eszközt és fordítja le a híreket."""
    fx_summary = fx_context["summary"] if fx_context else "EUR/HUF adat nem elérhető."

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
Gyakorlatias, döntésorientált befektetési tanácsadóként viselkedsz egy magyar befektető számára.
Elemezd az alábbi összes portfólióeszközt EGYETLEN válaszban.

--- DEVIZAKÖRNYEZET (FORINT HATÁS) ---
{fx_summary}
Szabály: Ha az eszköz devizás (EUR vagy USD) és a forint gyenge zónában van, a forintra történő profitrealizálás (részleges kitárazás) kedvezőbb, az új devizavétel drágább. Forintos eszközöknél (.BD vagy HUF) a saját trendre és hírekre fókuszálj.

--- ESZKÖZÖK LISTÁJA ---
{json.dumps(assets_payload, ensure_ascii=False)}

Feladat minden egyes "id" kulcshoz:
1. Dönts külön a MEGLÉVŐ pozícióról és az ÚJ vásárlásról.
2. Fordítsd le magyarra a kapott angol/idegen nyelvű hírcímeket ("headlines").

Válaszolj KIZÁRÓLAG érvényes JSON objektummal, ahol a legfelső szintű kulcsok az eszközök "id" értékei ("A0", "A1", ...), és minden eszköz alatt pontosan ezek a kulcsok szerepelnek:
- "existing_portfolio_action": pontosan egy ezekből: "TARTÁS", "HÍGÍTÁS (DCA)", "RÉSZLEGES KITÁRAZÁS", "TELJES KITÁRAZÁS"
- "new_entry_action": pontosan egy ezekből: "ÉRDEMES VENNI", "KIVÁRÁS", "KERÜLENDŐ"
- "plain_tech_summary": 1 rövid magyar mondat szakszavak (RSI/SMA) nélkül arról, hol áll az árfolyam
- "sentiment_score": egész szám 1 és 100 között
- "reasoning": 2 tömör magyar mondat arról, miért ezt javaslod most
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


def render_technical_learning_panel(tech):
    """Látható SMA 50 / SMA 200 / RSI értékelő és szabályalapú tanuló magyarázat."""
    if not tech:
        return

    price = tech.get("price")
    sma50 = tech.get("sma_50")
    sma200 = tech.get("sma_200")
    rsi = tech.get("rsi_14")

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
        if rsi < 30:
            rsi_zone = "Túladott (<30) • Hígítási zóna"
        elif rsi > 70:
            rsi_zone = "Túlvett (>70) • Túlfűtött"
        else:
            rsi_zone = "Semleges zóna (30–70)"
        t3.metric("RSI (14 napos)", f"{rsi:.1f}", rsi_zone)
    else:
        t3.metric("RSI (14 napos)", "N/A", "Kevés adat (<15 nap)")

    rules_notes = []

    if price is not None and sma50 is not None and sma200 is not None:
        if price > sma50 and sma50 > sma200:
            rules_notes.append(
                "✅ **Trend (SMA 50 / SMA 200):** Az árfolyam az SMA 50 felett, az SMA 50 pedig az SMA 200 felett halad (erős emelkedő trend / Aranykereszt) → **Meglévő pozíció: TARTÁS**."
            )
        elif price < sma50 and sma50 < sma200:
            rules_notes.append(
                "🔴 **Trend (SMA 50 / SMA 200):** Az árfolyam mindkét mozgóátlag alatt van, és az SMA 50 az SMA 200 alá esett (tartós csökkenő trend) → **Meglévő pozíció: KITÁRAZÁS / KOCKÁZATCSÖKKENTÉS**, új vétel: **KERÜLENDŐ**."
            )
        elif price < sma50 and price >= sma200:
            rules_notes.append(
                "⚖️ **Trend (SMA 50 / SMA 200):** Az árfolyam beesett az SMA 50 alá, de az SMA 200 még tartja (átmeneti korrekció emelkedő főtrendben) → **Meglévő pozíció: TARTÁS**, új vétel: **KIVÁRÁS** (támasz teszteléséig)."
            )
        else:
            rules_notes.append(
                "🟡 **Trend (SMA 50 / SMA 200):** Az árfolyam az SMA 200 közelében/alatt billeg, iránykeresés zajlik → **KIVÁRÁS** a trend megerősítéséig."
            )
    elif price is not None and sma50 is not None:
        if price >= sma50:
            rules_notes.append("✅ **Középtávú trend:** Az árfolyam az SMA 50 felett van → **TARTÁS**.")
        else:
            rules_notes.append("⚠️ **Középtávú trend:** Az árfolyam az SMA 50 alatt van → **KIVÁRÁS**.")

    if rsi is not None:
        if rsi < 30:
            rules_notes.append(
                f"🟢 **Lendület (RSI = {rsi:.1f}):** 30 alatti túladott zóna – statisztikailag gyakori a felpattanás → **HÍGÍTÁS (DCA) / ÉRDEMES VENNI** (ha a főtrend nem omlott össze)."
            )
        elif rsi > 70:
            rules_notes.append(
                f"🟠 **Lendület (RSI = {rsi:.1f}):** 70 feletti túlvett zóna – rövid távon kifulladhat az emelkedés → **Új vétel: KIVÁRÁS**, meglévőnél **RÉSZLEGES KITÁRAZÁS** mérlegelhető."
            )
        else:
            rules_notes.append(
                f"🔵 **Lendület (RSI = {rsi:.1f}):** 30 és 70 közötti egészséges, kiegyensúlyozott tartomány → Nincs extrém beszállási vagy kiszállási kényszer (**TARTÁS / KIVÁRÁS**)."
            )

    if rules_notes:
        st.info("\n\n".join(rules_notes))

    with st.expander("📘 Gyors szabály-puska: Mikor mi a jelzés?"):
        st.markdown(
            """
            * **TARTÁS:** Az árfolyam az **SMA 50** és az **SMA 200** felett halad, az RSI pedig **30–70** között mozog.
            * **HÍGÍTÁS (DCA) / ÉRDEMES VENNI:** Az **RSI 30 alá** esik (túladott pánikzóna), vagy az árfolyam emelkedő főtrendben visszateszteli az **SMA 200** támaszt.
            * **KIVÁRÁS:** Az árfolyam az SMA 50 és SMA 200 között oldalazik, vagy az **RSI 65–70 felett** van (túl drága az új beszálláshoz).
            * **RÉSZLEGES / TELJES KITÁRAZÁS:** Az árfolyam tartósan az **SMA 200 alá** esik (trendtörés), vagy kiugró **RSI (>75)** mellett gyenge a forint (devizás eszközöknél).
            """
        )


def render_asset_details(item):
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

    render_technical_learning_panel(tech)

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

    status_box.info("2/4 • EUR/HUF devizakörnyezet lekérése...")
    fx_context = fetch_eurhuf_context()
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
        return [], fx_context, datetime.now().strftime("%Y.%m.%d. %H:%M"), False, "Nincs aktív eszköz.", None

    progress_bar.progress(78)
    status_box.info(f"4/4 • Összesített AI elemzés és hírfordítás ({len(raw_items)} eszköz 1 csomagban)...")
    ai_map, ai_err = analyze_all_assets_batch(raw_items, fx_context)

    results = []
    for item in raw_items:
        ai_res = ai_map.get(item["id"])
        if not ai_res:
            prefix = "[Szabályalapú mód] " if ai_err else ""
            ai_res = generate_rule_based_fallback(item["tech"], item["currency"], fx_context, prefix)

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

    return results, fx_context, run_timestamp, log_ok, log_msg, ai_err


# --- FELÜLET (UI) ---
st.markdown(
    """
    <div class="main-banner">
        <h1>🔥 AI PIACI IRÁNYTŰ</h1>
        <p>Gyors portfólió-döntéstámogató • Magyar blue chipek, VIG alapok & EUR/HUF hatás</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚡ Vezérlőpult")
    if st.button("🔄 Frissítés most (Élő adat)", use_container_width=True):
        st.session_state.pop("cached_payload", None)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📧 E-mail Riasztások")
    alerts_enabled = st.toggle("Automatikus riasztás", value=True)
    st.caption("Jelzést küld **KITÁRAZÁS** javaslat vagy **erős hígítási zóna** esetén.")

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

# Munkamenet-szintű gyorsítótár (1 órás érvényességgel), hogy nézetváltáskor azonnali legyen
now_dt = datetime.now()
cached = st.session_state.get("cached_payload")
needs_refresh = (
    cached is None
    or (now_dt - cached["fetched_at"]) > timedelta(hours=1)
)

if needs_refresh:
    try:
        res_tuple = run_full_analysis_with_progress()
        st.session_state["cached_payload"] = {
            "fetched_at": now_dt,
            "payload": res_tuple,
        }
    except Exception as e:
        st.error(f"Hiba az adatok betöltésekor: {e}")
        st.stop()

data, fx_context, last_updated, log_ok, log_msg, ai_err = st.session_state["cached_payload"]["payload"]

if ai_err:
    if "429" in ai_err or "RESOURCE_EXHAUSTED" in ai_err.upper():
        st.warning(
            "ℹ️ **Ingyenes AI napi kvótalimit elérve:** Az alkalmazás nem akadt el, hanem automatikusan "
            "a beépített SMA 50 / SMA 200 / RSI szabálymotor alapján számolta ki az értékeléseket."
        )
    else:
        st.warning(f"⚠️ **AI kiegészítő megjegyzés** (szabályalapú mód aktív): `{ai_err}`")

if fx_context:
    st.markdown(f'<div class="fx-strip">💱 {fx_context["summary"]}</div>', unsafe_allow_html=True)

st.caption(f"⏱️ Frissítve: **{last_updated}** &nbsp;|&nbsp; 💾 {log_msg if log_ok else '⚠️ ' + log_msg}")

if not data:
    st.warning("Nincs 'IGEN' státuszú eszköz az 'Eszkozok' vagy 'VIG_Alapok' munkalapon.")
    st.stop()

# Riasztások ellenőrzése nézettől függetlenül
alert_map = {}
if alerts_enabled:
    for item in data:
        item_uid = f"{item['name']}_{item['ticker']}"
        sent, status_msg = check_and_send_alert(item)
        if status_msg:
            alert_map[item_uid] = (sent, status_msg)
            if sent:
                st.toast(f"Riasztás elküldve: {item['name']}", icon="🔥")

# Nézetváltó és gyors szűrő
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

# Szűrés alkalmazása
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

    # Kártyaváltás nélküli gyors részletnéző egy kiválasztott eszközhöz
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
            render_asset_details(selected_item)

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

                render_asset_details(item)
