import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from email.message import EmailMessage
import json
from pathlib import Path
import re
import smtplib
import ssl
import time
import urllib.parse
import urllib.request

from google import genai
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import streamlit as st
import yfinance as yf

# Ikon elérési útja (a market_analyzer.py mellett a repó gyökerében)
ICON_PATH = Path(__file__).parent / "ikon.png"

# --- KÖZPONTI KATEGÓRIA- ÉS SZÍNKONFIGURÁCIÓ ---
# Ha a kriptó színét metálkékről pinkre váltanád, írd át a "color" értéket "#EC4899"-re!
CATEGORY_CONFIG = {
    "STOCKS": {
        "sheet_names": ["Eszkozok"],
        "label": "Hagyományos tőzsde (Részvények)",
        "short_label": "Részvény",
        "icon": "📈",
        "color": "#FF6B00",      # Narancs
        "is_crypto": False,
    },
    "CRYPTO": {
        "sheet_names": ["Kriptovaluták", "Kriptovalutak"],
        "label": "Kriptovaluták",
        "short_label": "Kriptó",
        "icon": "🪙",
        "color": "#0EA5E9",      # Metálkék (Pink alternatíva: "#EC4899")
        "is_crypto": True,
    },
    "ETF": {
        "sheet_names": ["ETF-ek", "ETF"],
        "label": "ETF-ek",
        "short_label": "ETF",
        "icon": "🌐",
        "color": "#10B981",      # Smaragdzöld
        "is_crypto": False,
    },
    "FUNDS": {
        "sheet_names": ["VIG_alapok", "VIG_Alapok"],
        "label": "Befektetési jegyek (VIG alapok)",
        "short_label": "Befektetési jegy",
        "icon": "🏛️",
        "color": "#E11D48",      # VIG-piros
        "is_crypto": False,
    },
}

CATEGORY_ORDER = ["STOCKS", "CRYPTO", "ETF", "FUNDS"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def hex_to_rgba(hex_color, alpha):
    """HEX színkódból CSS rgba() értéket képez."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def get_icon_base64():
    """Base64 formátumba alakítja a logó ikont, ha megtalálható a repóban."""
    if ICON_PATH.exists():
        return base64.b64encode(ICON_PATH.read_bytes()).decode("utf-8")
    return None


def get_clean_gemini_key():
    """Kinyeri és megtisztítja a Gemini API kulcsot a felesleges szóközöktől/idézőjelektől."""
    raw_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or ""
    return str(raw_key).strip().strip('"').strip("'")


def _send_gemini_request(client, model_name, prompt):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text or ""


def call_gemini_single_batch(prompt):
    """Kötegelt kérést küld az ingyenes Gemini API-nak, 503-as terhelési csúcs esetén rövid újrapróbálkozással."""
    api_key = get_clean_gemini_key()
    if not api_key:
        raise RuntimeError("Hiányzik a GEMINI_API_KEY a Secrets beállításokból.")

    client = genai.Client(vertexai=False, api_key=api_key)
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
                        time.sleep(2 * (attempt + 1))
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


def find_worksheet_by_candidates(spreadsheet, candidate_names):
    """Megkeresi a munkalapot pontos név vagy kis-/nagybetű-független egyezés alapján."""
    try:
        all_sheets = spreadsheet.worksheets()
    except Exception:
        return None

    for cand in candidate_names:
        for ws in all_sheets:
            if ws.title.strip() == cand:
                return ws

    cand_lower = [c.lower() for c in candidate_names]
    for ws in all_sheets:
        if ws.title.strip().lower() in cand_lower:
            return ws

    return None


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


def load_recent_alert_keys(spreadsheet):
    """Beolvassa az elmúlt 24 órában már elküldött riasztások kulcsait a 'Riasztasok' munkalapról."""
    headers = ["Időpont", "Kulcs", "Eszköz", "Jelzés"]
    try:
        ws = spreadsheet.worksheet("Riasztasok")
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title="Riasztasok", rows=1000, cols=len(headers))
        ws.append_row(headers, value_input_option="USER_ENTERED")
        return ws, set()

    rows = ws.get_all_values()
    if not rows:
        ws.append_row(headers, value_input_option="USER_ENTERED")
        return ws, set()

    recent_keys = set()
    cutoff = datetime.now() - timedelta(hours=24)
    for row in rows[1:]:
        if len(row) >= 2 and row[0] and row[1]:
            try:
                sent_dt = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
                if sent_dt >= cutoff:
                    recent_keys.add(row[1].strip())
            except ValueError:
                continue
    return ws, recent_keys


def process_alerts_batch(spreadsheet, results):
    """
    Ellenőrzi az összes eredményt, e-mailt küld az új jelzésekről, és naplózza őket a 'Riasztasok' fülre,
    így sem a háttérfuttató, sem a kézi frissítés nem küld dupla levelet 24 órán belül.
    """
    if not results:
        return {}

    try:
        alert_ws, recent_keys = load_recent_alert_keys(spreadsheet)
    except Exception:
        alert_ws, recent_keys = None, set()

    alert_map = {}
    new_alert_rows = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for item in results:
        existing_act = item.get("existing_action", "").upper()
        new_act = item.get("new_entry_action", "").upper()
        tech = item.get("tech") or {}
        rsi = tech.get("rsi_14")
        cat_cfg = CATEGORY_CONFIG.get(item.get("category", "STOCKS"), CATEGORY_CONFIG["STOCKS"])
        cat_color = cat_cfg["color"]
        item_uid = f"{item['category']}_{item['name']}_{item['ticker']}"

        triggers = []
        if "KITÁRAZÁS" in existing_act:
            triggers.append(f"Portfólió jelzés: {item['existing_action']}")
        if "ÉRDEMES VENNI" in new_act or "HÍGÍTÁS" in existing_act:
            rsi_txt = f" (RSI: {rsi:.1f})" if rsi is not None else ""
            triggers.append(f"Kedvező beszállási / hígítási zóna{rsi_txt}")

        if not triggers:
            continue

        alert_key = f"{item['category']}_{item['name']}_{item['ticker']}_{'_'.join(triggers)}"
        if alert_key in recent_keys:
            alert_map[item_uid] = (False, "Már riasztva (24h)")
            continue

        trigger_text = " | ".join(triggers)
        price_str = f"{tech['price']:,.4f}" if tech.get("price") is not None else "N/A"
        change_str = f"{tech['change_pct']:+.2f}%" if tech.get("change_pct") is not None else "N/A"

        subject = f"🔥 Piaci Jelzés [{cat_cfg['short_label']}]: {item['name']} ({item['ticker']}) – {item['existing_action']}"

        plain_body = (
            f"Piaci jelzés érkezett:\n\n"
            f"Kategória: {cat_cfg['label']}\n"
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
            <div style="border-left: 5px solid {cat_color}; padding-left: 14px;">
              <div style="font-size: 12px; font-weight: bold; color: {cat_color}; text-transform: uppercase;">{cat_cfg['label']}</div>
              <h2 style="color: {cat_color}; margin: 2px 0 4px 0;">🔥 {item['name']} ({item['ticker']})</h2>
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
            recent_keys.add(alert_key)
            alert_map[item_uid] = (True, "Riasztás elküldve")
            new_alert_rows.append([now_str, alert_key, f"{item['name']} ({item['ticker']})", trigger_text])
        except Exception as e:
            alert_map[item_uid] = (False, f"SMTP hiba: {e}")

    if alert_ws is not None and new_alert_rows:
        try:
            alert_ws.append_rows(new_alert_rows, value_input_option="USER_ENTERED")
        except Exception:
            pass

    return alert_map


def _fetch_single_fx_pair(ticker_symbol, pair_label):
    """Lekéri egy konkrét devizapár (pl. EUR/HUF vagy USD/HUF) 1 éves pozícióját és devizafelárát."""
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
        avg_1y = float(close.mean())
        low_1y = float(close.min())
        high_1y = float(close.max())
        pos_pct = ((rate - low_1y) / (high_1y - low_1y) * 100) if high_1y > low_1y else 50.0
        premium_vs_low_pct = ((rate - low_1y) / low_1y * 100) if low_1y > 0 else 0.0
        premium_vs_avg_pct = ((rate - avg_1y) / avg_1y * 100) if avg_1y > 0 else 0.0
        base_curr = pair_label.split("/")[0]

        if pos_pct >= 75:
            regime_short = (
                f"Erős {base_curr} / Gyenge HUF zóna "
                f"(+{premium_vs_low_pct:.1f}% az 1 éves mélyponthoz képest)"
            )
            zone_tag = "WEAK_HUF"
        elif pos_pct <= 25:
            regime_short = (
                f"Gyenge {base_curr} / Erős HUF zóna "
                f"(olcsó {base_curr} – devizaváltásra és új vételre kedvező)"
            )
            zone_tag = "STRONG_HUF"
        else:
            regime_short = f"Semleges {pair_label} zóna (középmezőnyben mozgó árfolyam)"
            zone_tag = "NEUTRAL_HUF"

        return {
            "pair": pair_label,
            "rate": rate,
            "change_pct": chg,
            "sma50": sma50,
            "low_1y": low_1y,
            "high_1y": high_1y,
            "pos_pct": pos_pct,
            "premium_vs_low_pct": premium_vs_low_pct,
            "premium_vs_avg_pct": premium_vs_avg_pct,
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


def get_ticker_assets_from_sheet(spreadsheet, category_key):
    """Beolvassa az 'IGEN' státuszú ticker alapú eszközöket az adott kategória munkalapjáról."""
    cfg = CATEGORY_CONFIG[category_key]
    ws = find_worksheet_by_candidates(spreadsheet, cfg["sheet_names"])
    if ws is None:
        return []

    records = ws.get_all_records()
    items = []
    for row in records:
        if str(row.get("aktiv", "")).strip().upper() == "IGEN" and row.get("ticker"):
            ticker = str(row["ticker"]).strip()
            name = str(row.get("name", ticker)).strip() or ticker
            items.append({
                "name": name,
                "ticker": ticker,
                "category": category_key,
            })
    return items


# --- BAMOSZ AUTOMATIKUS ÁRFOLYAMLETÖLTÉS (ISIN ALAPJÁN) ---
def _parse_hu_float(val_str):
    """Magyar és nemzetközi számformátumok biztonságos átalakítása."""
    s = str(val_str).strip().replace("\xa0", "").replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        val = float(s)
        return val if val > 0 else None
    except ValueError:
        return None


def _parse_hu_date(date_str):
    """Magyar ('2026.11.20.') és ISO dátumok átalakítása."""
    s = str(date_str).strip().rstrip(".")
    s = re.sub(r"\s+", "", s).replace("/", "-").replace(".", "-")
    dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce")
    return dt


def _http_get_text(url, timeout=12):
    """Böngésző fejléccel letölti az adott URL szöveges tartalmát."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,text/csv,application/json,*/*;q=0.8",
            "Accept-Language": "hu-HU,hu;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_bytes = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        try:
            return raw_bytes.decode(charset, errors="replace")
        except LookupError:
            return raw_bytes.decode("utf-8", errors="replace")


def _extract_points_from_bamosz_payload(text):
    """Kinyeri a (dátum, árfolyam) párokat BAMOSZ CSV-ből, beágyazott grafikon-adatokból vagy HTML táblából."""
    points = {}

    for y_s, m_s, d_s, val_s in re.findall(
        r"Date\.UTC\(\s*(20\d{2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)\s*,\s*([0-9]+(?:\.[0-9]+)?)",
        text,
    ):
        try:
            dt = pd.Timestamp(year=int(y_s), month=int(m_s) + 1, day=int(d_s))
            val = float(val_s)
            if val > 0:
                points[dt] = val
        except ValueError:
            continue

    if len(points) >= 10:
        return points

    for ms_s, val_s in re.findall(r"\[\s*(1[678]\d{11})\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]", text):
        try:
            dt = pd.to_datetime(int(ms_s), unit="ms", errors="coerce")
            val = float(val_s)
            if pd.notna(dt) and val > 0:
                points[dt.normalize()] = val
        except ValueError:
            continue

    if len(points) >= 10:
        return points

    for line in text.splitlines():
        clean_line = line.strip()
        if not clean_line or "<" in clean_line:
            continue
        parts = re.split(r"[;\t,]", clean_line)
        if len(parts) >= 2:
            dt = _parse_hu_date(parts[0].strip('"').strip("'"))
            if pd.notna(dt) and 2000 <= dt.year <= 2035:
                for col_val in parts[1:4]:
                    val = _parse_hu_float(col_val.strip('"').strip("'"))
                    if val is not None:
                        points[dt.normalize()] = val
                        break

    if len(points) >= 5:
        return points

    for tr_html in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I):
        cells = [
            re.sub(r"<[^>]+>", "", td).strip()
            for td in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr_html, flags=re.S | re.I)
        ]
        if len(cells) < 2:
            continue
        found_dt = None
        found_val = None
        for cell in cells:
            if found_dt is None and re.search(r"20\d{2}[.\-/]\s*\d{1,2}[.\-/]\s*\d{1,2}", cell):
                dt_cand = _parse_hu_date(cell)
                if pd.notna(dt_cand):
                    found_dt = dt_cand.normalize()
                    continue
            if found_dt is not None and found_val is None:
                if "%" in cell or re.search(r"[A-Za-z]{3,}", cell):
                    continue
                if re.match(r"^\d[\d\s.,]*\d$|^\d+(?:[.,]\d+)?$", cell.strip()):
                    val_cand = _parse_hu_float(cell)
                    if val_cand is not None:
                        found_val = val_cand
                        break
        if found_dt is not None and found_val is not None:
            points[found_dt] = found_val

    return points


def fetch_bamosz_history(isin_code):
    """ISIN kód alapján letölti az alap visszamenőleges napi árfolyamsorát a BAMOSZ oldaláról."""
    isin = str(isin_code or "").strip().upper()
    if not re.match(r"^[A-Z]{2}[A-Z0-9]{10}$", isin):
        return None, "Érvénytelen ISIN formátum"

    candidate_urls = [
        f"https://www.bamosz.hu/alapoldal?isin={urllib.parse.quote(isin)}",
        f"https://www.bamosz.hu/hu/alapoldal?isin={urllib.parse.quote(isin)}",
    ]

    all_points = {}
    last_err = None

    for base_url in candidate_urls:
        try:
            html_text = _http_get_text(base_url, timeout=12)
            pts = _extract_points_from_bamosz_payload(html_text)
            all_points.update(pts)

            export_links = re.findall(
                r'href=["\']([^"\']*(?:csv|export|letoltes)[^"\']*)["\']',
                html_text,
                flags=re.I,
            )
            for rel_link in export_links[:2]:
                full_export_url = urllib.parse.urljoin(base_url, rel_link.replace("&amp;", "&"))
                try:
                    csv_text = _http_get_text(full_export_url, timeout=12)
                    csv_pts = _extract_points_from_bamosz_payload(csv_text)
                    all_points.update(csv_pts)
                except Exception:
                    pass

            if len(all_points) >= 5:
                break
        except Exception as e:
            last_err = e

    if not all_points:
        err_hint = f" ({last_err})" if last_err else ""
        return None, f"BAMOSZ adat nem érhető el{err_hint}"

    sorted_dates = sorted(all_points.keys())
    prices = [all_points[d] for d in sorted_dates]
    df = pd.DataFrame({"Close": prices}, index=pd.DatetimeIndex(sorted_dates))
    return df, f"BAMOSZ élő idősor ({len(df)} nap)"


def get_vig_funds_from_sheet(spreadsheet):
    """Beolvassa a 'VIG_alapok' munkalapot (ISIN alapú BAMOSZ letöltés + kézi tartalék)."""
    cfg = CATEGORY_CONFIG["FUNDS"]
    vig_sheet = find_worksheet_by_candidates(spreadsheet, cfg["sheet_names"])
    if vig_sheet is None:
        return []

    records = vig_sheet.get_all_records()
    grouped = {}

    for row in records:
        if str(row.get("aktiv", "IGEN")).strip().upper() != "IGEN":
            continue

        fund_name = str(row.get("alap_nev", "")).strip()
        if not fund_name:
            continue

        isin = str(row.get("isin", "")).strip().upper()
        currency = str(row.get("deviza", "HUF")).strip().upper() or "HUF"
        strategy = str(row.get("strategia", "")).strip()
        date_str = str(row.get("datum", "")).strip()
        raw_price = str(row.get("arfolyam", "")).strip()

        entry = grouped.setdefault(
            fund_name,
            {
                "isin": isin,
                "currency": currency,
                "strategy": strategy,
                "manual_points": [],
            },
        )
        if isin and not entry["isin"]:
            entry["isin"] = isin
        if strategy and not entry["strategy"]:
            entry["strategy"] = strategy
        if currency:
            entry["currency"] = currency

        if raw_price:
            price_val = _parse_hu_float(raw_price)
            if price_val is not None:
                entry["manual_points"].append((date_str, price_val))

    funds = []
    for fund_name, info in grouped.items():
        isin = info["isin"]
        currency = info["currency"]
        strategy = info["strategy"]

        bamosz_df = None
        source_note = "Kézi táblázat"

        if isin:
            bamosz_df, source_note = fetch_bamosz_history(isin)

        manual_df = None
        if info["manual_points"]:
            pts = sorted(info["manual_points"], key=lambda x: x[0])
            dates = pd.to_datetime([_parse_hu_date(p[0]) for p in pts], errors="coerce")
            prices = [p[1] for p in pts]
            idx = dates if not pd.isna(dates).all() else range(len(prices))
            manual_df = pd.DataFrame({"Close": prices}, index=idx)

        if bamosz_df is not None and not bamosz_df.empty:
            final_df = bamosz_df
            if manual_df is not None and isinstance(manual_df.index, pd.DatetimeIndex):
                combined = pd.concat([bamosz_df, manual_df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                final_df = combined
        elif manual_df is not None and not manual_df.empty:
            final_df = manual_df
            if isin:
                source_note = f"Kézi tartalék ({source_note})"
        else:
            final_df = pd.DataFrame(columns=["Close"])

        ticker_label = f"VIG | {isin}" if isin else f"VIG ({currency})"
        funds.append({
            "name": fund_name,
            "ticker": ticker_label,
            "isin": isin,
            "currency": currency,
            "strategy": strategy,
            "source_note": source_note,
            "category": "FUNDS",
            "hist_df": final_df,
        })

    return funds


def log_results_to_sheet(spreadsheet, results, run_timestamp, trigger_source="Kézi / UI"):
    """Kötegelve menti az elemzési eredményeket a Google Táblázat 'Naplo' munkalapjára."""
    if not results:
        return False, "Nincs menthető adat."

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
        cat_short = CATEGORY_CONFIG.get(item.get("category", "STOCKS"), CATEGORY_CONFIG["STOCKS"])["short_label"]
        price_val = round(tech["price"], 4) if tech.get("price") is not None else ""
        change_val = round(tech["change_pct"], 2) if tech.get("change_pct") is not None else ""
        rsi_val = round(tech["rsi_14"], 2) if tech.get("rsi_14") is not None else ""
        sma50_val = round(tech["sma_50"], 4) if tech.get("sma_50") is not None else ""
        sma200_val = round(tech["sma_200"], 4) if tech.get("sma_200") is not None else ""

        rows_to_append.append([
            f"{run_timestamp} ({trigger_source})",
            f"[{cat_short}] {item['name']}",
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
    """Kiszámolja a 14 napos RSI-t, az 50/200 napos mozgóátlagot és az 52 hetes volatilitási sávot."""
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

    close_1y = close.tail(252)
    low_52w = float(close_1y.min())
    high_52w = float(close_1y.max())
    range_52w_pct = ((high_52w - low_52w) / low_52w * 100) if low_52w > 0 else None

    ann_vol_pct = None
    if len(close_1y) >= 20:
        daily_ret = close_1y.pct_change().dropna()
        if len(daily_ret) >= 15:
            ann_vol_pct = float(daily_ret.std() * (252 ** 0.5) * 100)

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
        "low_52w": low_52w,
        "high_52w": high_52w,
        "range_52w_pct": range_52w_pct,
        "ann_vol_pct": ann_vol_pct,
        "trend": trend,
        "chart_df": chart_df,
    }


def evaluate_technical_matrix(tech, currency="USD", fx_map=None, category="STOCKS"):
    """KATEGÓRIA- ÉS VOLATILITÁSFÜGGŐ DÖNTÉSI MÁTRIX (Trend × RSI × Súlyozott Devizapár)."""
    fx_map = fx_map or {}
    cat_cfg = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["STOCKS"])
    is_crypto = cat_cfg.get("is_crypto", False)

    relevant_fx = fx_map.get(currency) if currency in ("EUR", "USD") else None
    zone_tag = relevant_fx.get("zone_tag") if relevant_fx else None
    weak_huf = zone_tag == "WEAK_HUF"
    strong_huf = zone_tag == "STRONG_HUF"

    range_52w = tech.get("range_52w_pct") if tech else None
    fx_prem = relevant_fx.get("premium_vs_low_pct", 0.0) if relevant_fx else 0.0
    fx_to_swing_pct = (fx_prem / range_52w * 100) if (relevant_fx and range_52w and range_52w > 0) else None

    if relevant_fx:
        swing_info = ""
        if range_52w is not None and fx_to_swing_pct is not None:
            swing_info = (
                f" Az eszköz 52 hetes mozgástere **{range_52w:.0f}%**, így a **{fx_prem:.1f}%-os** devizafelár "
                f"ennek csak **{fx_to_swing_pct:.0f}%-a**."
            )
        if is_crypto and weak_huf:
            fx_note = (
                f"💱 **Devizahatás – Kriptó súlyozás ({relevant_fx['pair']} = {relevant_fx['rate']:.2f} Ft):** "
                f"A {currency} ugyan magas ({fx_prem:+.1f}% az 1 éves mélyponthoz képest), de a kriptoeszköz nagy "
                f"volatilitása és éves felpattanási potenciálja mellett ez **másodlagos tényező**."
                f"{swing_info}"
            )
        elif weak_huf:
            fx_note = (
                f"💱 **Devizahatás – Magas súly ({relevant_fx['pair']} = {relevant_fx['rate']:.2f} Ft):** "
                f"{relevant_fx['regime_short']} — hagyományos eszköznél ez érdemben drágítja az új HUF-alapú vételt."
                f"{swing_info}"
            )
        else:
            fx_note = (
                f"💱 **Devizahatás ({relevant_fx['pair']} = {relevant_fx['rate']:.2f} Ft):** "
                f"{relevant_fx['regime_short']} (1 éves sáv {relevant_fx['pos_pct']:.0f}%-ánál)."
                f"{swing_info}"
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
            "plain_summary": "Nincs elegendő historikus árfolyamadat (ellenőrizd az ISIN kódot).",
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

    if trend_state in ("UP_STRONG", "UP_PULLBACK"):
        if rsi_state in ("OVERSOLD_DEEP", "OVERSOLD_NEAR"):
            existing_act = "HÍGÍTÁS (DCA)"
            new_act = "ÉRDEMES VENNI"
            score = 84 if strong_huf else (80 if is_crypto else 76)
            if strong_huf and relevant_fx:
                fx_extra = (
                    f" Ráadásul Gyenge {currency} / Erős HUF ({relevant_fx['rate']:.2f} Ft) zóna van, "
                    "így a devizaváltás is olcsó."
                )
            elif weak_huf and relevant_fx and is_crypto:
                fx_extra = (
                    f" Bár a {currency} magas ({relevant_fx['rate']:.2f} Ft), a kriptoeszköz nagy éves "
                    "mozgásteréhez képest ez másodlagos, így nem akadálya a szakaszos vételnek."
                )
            elif weak_huf and relevant_fx:
                fx_extra = (
                    f" Figyelem: Erős {currency} / Gyenge HUF ({relevant_fx['rate']:.2f} Ft) zóna miatt "
                    f"drága a {currency}, így forintból váltva csak óvatos részletekben érdemes belépni."
                )
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
                    f"Túlvett RSI ({rsi:.1f}) mellett Erős {currency} / Gyenge HUF ({relevant_fx['rate']:.2f} Ft) zónában jár az árfolyam: "
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
                fx_extra = (
                    f" (A Gyenge {currency} / Erős HUF árfolyam miatt devizát előre váltani kedvező, "
                    "de a belépéssel érdemes támaszra várni.)"
                )
            elif weak_huf and relevant_fx and is_crypto:
                fx_extra = (
                    f" (A {currency} árfolyama magas, de kriptónál ez másodlagos; inkább az RSI lehűlésére érdemes várni.)"
                )
            elif weak_huf and relevant_fx:
                fx_extra = (
                    f" (Ráadásul Erős {currency} / Gyenge HUF zóna miatt forintból váltani most drága.)"
                )
            else:
                fx_extra = ""
            reason = (
                "Az emelkedő főtrend ép, de az RSI semleges tartományban mozog (nincs túladott diszkont): "
                f"meglévő pozíciót érdemes tartani, új vásárlással pedig kivárni.{fx_extra}"
            )
            matrix_note = "🎯 **Kombinált döntés (Emelkedő trend + Semleges RSI):** Meglévő pozícióra **TARTÁS**, új vételre **KIVÁRÁS**."

    elif trend_state in ("DOWN_BROKEN", "DOWN_CONFIRMED"):
        if rsi_state in ("OVERSOLD_DEEP", "OVERSOLD_NEAR"):
            existing_act = "TARTÁS"
            new_act = "KIVÁRÁS"
            score = 46 if is_crypto else 42
            if is_crypto and weak_huf and relevant_fx:
                fx_crypto_note = (
                    f" Bár az USD/HUF magas ({relevant_fx['rate']:.2f} Ft), a kriptó nagy éves mozgásteréhez "
                    "képest ez másodlagos — hosszú távú építkezéshez az RSI 35 fölé fordulásakor már "
                    "indokolt lehet a szakaszos (részletekben történő) vétel."
                )
            else:
                fx_crypto_note = " Új vétellel a hulló kés miatt ki kell várni az RSI 35 fölé fordulását."
            reason = (
                f"Bár a rövid távú RSI ({rsi:.1f}) már túladott (eladni a pánik alján már késő, jöhet felpattanás), "
                f"az árfolyam az SMA 200 alatt van.{fx_crypto_note}"
            )
            matrix_note = "🎯 **Kombinált döntés (SMA 200 alatt + Alacsony RSI = Hulló kés):** Pánikból eladni már késő (**Meglévő: TARTÁS**), új pénzzel meg kell várni az RSI 35 fölé fordulását (**Új vétel: KIVÁRÁS**)."
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
    """EGYETLEN kötegelt Gemini API hívással elemzi az összes eszközt kategória-súlyozott devizalogikával."""
    eur_summary = fx_map["EUR"]["summary"] if fx_map.get("EUR") else "EUR/HUF adat nem elérhető."
    usd_summary = fx_map["USD"]["summary"] if fx_map.get("USD") else "USD/HUF adat nem elérhető."

    assets_payload = []
    for item in raw_items:
        tech = item["tech"]
        cat_cfg = CATEGORY_CONFIG.get(item.get("category", "STOCKS"), CATEGORY_CONFIG["STOCKS"])
        if tech:
            rsi_str = f"{tech['rsi_14']:.1f}" if tech["rsi_14"] is not None else "N/A"
            sma50_str = f"{tech['sma_50']:,.4f}" if tech["sma_50"] is not None else "N/A"
            sma200_str = f"{tech['sma_200']:,.4f}" if tech["sma_200"] is not None else "N/A"
            range_str = f"{tech['range_52w_pct']:.0f}%" if tech.get("range_52w_pct") is not None else "N/A"
            tech_txt = (
                f"Ár: {tech['price']:,.4f} ({tech['change_pct']:+.2f}%), "
                f"RSI14: {rsi_str}, SMA50: {sma50_str}, SMA200: {sma200_str}, "
                f"52 hetes mozgástér: {range_str}, Trend: {tech['trend']}"
            )
        else:
            tech_txt = "Nincs technikai adat."

        entry = {
            "id": item["id"],
            "category": cat_cfg["label"],
            "is_crypto": cat_cfg["is_crypto"],
            "name": item["name"],
            "ticker": item["ticker"],
            "currency": item["currency"],
            "technical": tech_txt,
            "headlines": [n["title"] for n in item["raw_news"]],
        }
        if item.get("strategy"):
            entry["fund_strategy"] = item["strategy"]
        assets_payload.append(entry)

    prompt = f"""
Gyakorlatias, kockázattudatos befektetési tanácsadóként viselkedsz egy magyar befektető számára.
Elemezd az alábbi összes portfólióeszközt EGYETLEN válaszban.

--- DEVIZAKÖRNYEZET ÉS KATEGÓRIA-SÚLYOZOTT ÉRTELMEZÉSE (NAGYON FONTOS!) ---
- EUR eszközökhöz: {eur_summary}
- USD eszközökhöz: {usd_summary}
- HUF eszközökhöz: devizahatás nem értelmezhető, kizárólag a saját trendre és alapstratégiára/hírekre fókuszálj.
- ALAPVETŐ DEVIZA-SZABÁLY:
  - "Erős USD / Gyenge HUF" (vagy "Erős EUR / Gyenge HUF") = a deviza DRÁGA forintban nézve! Soha ne írd ilyenkor, hogy a deviza gyenge vagy olcsó!
  - "Gyenge USD / Erős HUF" (vagy "Gyenge EUR / Erős HUF") = a deviza OLCSÓ forintban nézve.
- KATEGÓRIA- ÉS VOLATILITÁSFÜGGŐ SÚLYOZÁS:
  - **Hagyományos részvények, ETF-ek és befektetési jegyek (`is_crypto: false`) esetén:** Egy 10–13%-os devizafelár (Erős USD/EUR / Gyenge HUF) az éves hozampotenciál jelentős részét kiteszi, ezért **szigorúan bünteti** az új forintalapú vételt (drága deviza miatt óvatosságra/kivárásra ints). Befektetési alapoknál vedd figyelembe a `fund_strategy` mezőben leírt mandátumot is (pl. aktív vegyes vagy abszolút hozamú alap).
  - **Kriptovaluták (`is_crypto: true`) esetén:** A kriptók 52 hetes árfolyam-kilengése (gyakran 100–200%+) többszöröse a devizakilengésnek. Ha az USD/HUF 10% körül drágább, kifejezetten jelezd az indoklásban, hogy bár a forint gyengesége drágítja a belépőt, **a kripto saját történelmi ugrásaihoz és volatilitásához képest ez másodlagos súlyú** — így erős támasznál vagy mélyen túladott RSI-nél nem zárja ki a szakaszos (részletekben történő) vételt!

--- DÖNTÉSI SZABÁLYOK (SZIGORÚAN TARTSD BE MIND az 5 PONTOT!) ---
1. Emelkedő trendben (Ár > SMA 200) ÉS alacsony RSI (< 35) mellett: Meglévő = "HÍGÍTÁS (DCA)", Új vétel = "ÉRDEMES VENNI". (Részvénynél/ETF-nél/alapnál figyelmeztess az erős USD/EUR devizafelárra; kriptónál jelezd, hogy a devizafelár a kripto volatilitása mellett másodlagos.)
2. Emelkedő trendben (Ár > SMA 200) ÉS semleges vagy emelkedett RSI (35 <= RSI <= 70) mellett: Meglévő = "TARTÁS", Új vétel = "KIVÁRÁS". Kisebb napi/heti visszahúzódás önmagában NEM indokol vételt vagy hígítást, amíg az RSI 35 felett van!
3. Megtört főtrendben (Ár < SMA 200) ÉS alacsony RSI (< 32) mellett ("hulló kés"): pánikból eladni már késő (Meglévő = "TARTÁS"), új vételhez meg kell várni az RSI 35 fölé fordulását (Új vétel = "KIVÁRÁS"). Kriptónál emeld ki, hogy a magas USD/HUF másodlagos súlyú, és az RSI fordulásakor szakaszosan (részletekben) érdemes lehet elkezdeni a vételt.
4. Tartós csökkenő trendben (Ár < SMA 50 < SMA 200) ÉS nem túladott RSI (>= 32) mellett: Meglévő = "RÉSZLEGES KITÁRAZÁS", Új vétel = "KERÜLENDŐ".
5. Túlvett RSI (> 70) esetén: ha az adott eszköz devizájánál "Erős USD / Gyenge HUF" vagy "Erős EUR / Gyenge HUF" zóna van, javasolj "RÉSZLEGES KITÁRAZÁS"-t forintra, különben "TARTÁS"-t; új vételre mindenképp "KIVÁRÁS"-t.

--- ESZKÖZÖK LISTÁJA ---
{json.dumps(assets_payload, ensure_ascii=False)}

Feladat minden egyes "id" kulcshoz:
1. Dönts külön a MEGLÉVŐ pozícióról és az ÚJ vásárlásról a fenti szabályok alapján.
2. Fordítsd le magyarra a kapott angol/idegen nyelvű hírcímeket ("headlines").

Válaszolj KIZÁRÓLAG érvényes JSON objektummal, ahol a legfelső szintű kulcsok az eszközök "id" értékei ("A0", "A1", ...), és minden eszköz alatt pontosan ezek a kulcsok szerepelnek:
- "existing_portfolio_action": pontosan egy ezekből: "TARTÁS", "HÍGÍTÁS (DCA)", "RÉSZLEGES KITÁRAZÁS", "TELJES KITÁRAZÁS"
- "new_entry_action": pontosan egy ezekből: "ÉRDEMES VENNI", "KIVÁRÁS", "KERÜLENDŐ"
- "plain_tech_summary": 1 rövid magyar mondat szakszavak (RSI/SMA) nélkül arról, hol áll az árfolyam
- "sentiment_score": egész szám 1 és 100 között
- "reasoning": 2 tömör magyar mondat arról, miért ezt javaslod most (sződd bele a híreket/alapstratégiát és a kategória-súlyozott "Erős/Gyenge USD/EUR" devizahatást is)
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
    """Meghatározza a döntési doboz színét a jelzés alapján."""
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


def run_core_analysis(progress_cb=None, trigger_source="Kézi / UI", send_alerts=True):
    """
    Közös elemzőmotor a Streamlit UI és a 0-24 órás GitHub Actions háttérfuttató számára.
    progress_cb: opcionális visszahívás (pct: int, msg: str) a felületi folyamatjelzőhöz.
    """
    def notify(pct, msg):
        if progress_cb is not None:
            progress_cb(pct, msg)

    notify(5, "1/4 • Csatlakozás a Google Táblázat 4 kategóriafüléhez...")
    client = get_gspread_client()
    spreadsheet = client.open_by_key(st.secrets["SHEET_ID"])

    ticker_assets = []
    for cat_key in ["STOCKS", "CRYPTO", "ETF"]:
        ticker_assets.extend(get_ticker_assets_from_sheet(spreadsheet, cat_key))

    notify(12, "1/4 • VIG alapok és BAMOSZ árfolyamsorok lekérése...")
    vig_funds = get_vig_funds_from_sheet(spreadsheet)
    notify(18, "2/4 • EUR/HUF és USD/HUF devizakörnyezet lekérése...")

    fx_map = fetch_fx_context()
    notify(28, "3/4 • Árfolyamok és hírek letöltése...")

    raw_items = []
    total_count = len(ticker_assets) + len(vig_funds)

    for idx, asset in enumerate(ticker_assets):
        ticker = asset["ticker"]
        name = asset["name"]
        category = asset["category"]
        currency = infer_currency_from_ticker(ticker)
        cat_short = CATEGORY_CONFIG[category]["short_label"]

        if total_count > 0:
            pct = 28 + int((idx / total_count) * 42)
            notify(
                min(70, pct),
                f"3/4 • Árfolyamok letöltése ({idx + 1}/{total_count}) [{cat_short}]: **{name}** ({ticker})...",
            )

        tech_data, news_items = fetch_yahoo_data(ticker, name)
        raw_items.append({
            "id": f"A{len(raw_items)}",
            "category": category,
            "name": name,
            "ticker": ticker,
            "currency": currency,
            "strategy": "",
            "source_note": "Yahoo Finance",
            "tech": tech_data,
            "raw_news": news_items,
        })

    if vig_funds:
        vig_news = search_news_fallback("VIG Asset Management")
        for f_idx, fund in enumerate(vig_funds):
            step_num = len(ticker_assets) + f_idx + 1
            if total_count > 0:
                pct = 28 + int(((step_num - 1) / total_count) * 42)
                notify(
                    min(70, pct),
                    f"3/4 • VIG alap feldolgozása ({step_num}/{total_count}): **{fund['name']}** ({fund['source_note']})...",
                )

            tech_data = calculate_technical_indicators(fund["hist_df"])
            raw_items.append({
                "id": f"A{len(raw_items)}",
                "category": "FUNDS",
                "name": fund["name"],
                "ticker": fund["ticker"],
                "currency": fund["currency"],
                "strategy": fund.get("strategy", ""),
                "source_note": fund.get("source_note", "BAMOSZ"),
                "tech": tech_data,
                "raw_news": vig_news,
            })

    if not raw_items:
        return [], fx_map, datetime.now().strftime("%Y.%m.%d. %H:%M"), False, "Nincs aktív eszköz.", None, {}

    notify(78, f"4/4 • Összesített AI elemzés és hírfordítás ({len(raw_items)} eszköz 1 csomagban)...")
    ai_map, ai_err = analyze_all_assets_batch(raw_items, fx_map)

    results = []
    for item in raw_items:
        ai_res = ai_map.get(item["id"])
        if not ai_res:
            eval_res = evaluate_technical_matrix(
                item["tech"],
                item["currency"],
                fx_map,
                item["category"],
            )
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
            "category": item["category"],
            "name": item["name"],
            "ticker": item["ticker"],
            "currency": item["currency"],
            "strategy": item.get("strategy", ""),
            "source_note": item.get("source_note", ""),
            "tech": item["tech"],
            "existing_action": ai_res.get("existing_portfolio_action", "TARTÁS"),
            "new_entry_action": ai_res.get("new_entry_action", "KIVÁRÁS"),
            "plain_tech": ai_res.get("plain_tech_summary", ""),
            "score": ai_res.get("sentiment_score", 50),
            "reasoning": ai_res.get("reasoning", ""),
            "news_list": paired_news,
        })

    notify(90, "Naplózás a Google Táblázatba...")
    run_timestamp = datetime.now().strftime("%Y.%m.%d. %H:%M")

    try:
        log_ok, log_msg = log_results_to_sheet(spreadsheet, results, run_timestamp, trigger_source=trigger_source)
    except Exception as e:
        log_ok, log_msg = False, f"Naplózási hiba: {e}"

    alert_map = {}
    if send_alerts:
        notify(96, "Riasztások ellenőrzése...")
        alert_map = process_alerts_batch(spreadsheet, results)

    notify(100, "Kész!")
    return results, fx_map, run_timestamp, log_ok, log_msg, ai_err, alert_map


# --- STREAMLIT FELÜLETI ELEMEK (UI) ---
def render_technical_learning_panel(tech, currency="USD", fx_map=None, category="STOCKS"):
    """Látható SMA 50 / SMA 200 / RSI / 52 hetes mozgástér értékelő és tanuló magyarázat."""
    if not tech:
        return

    price = tech.get("price")
    sma50 = tech.get("sma_50")
    sma200 = tech.get("sma_200")
    rsi = tech.get("rsi_14")
    range_52w = tech.get("range_52w_pct")
    eval_res = evaluate_technical_matrix(tech, currency, fx_map, category)

    st.markdown("##### 📐 Technikai mutatók és döntési logika (Tanuló nézet)")

    t1, t2, t3, t4 = st.columns(4)

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

    if range_52w is not None:
        t4.metric("52 hetes kilengés", f"{range_52w:.0f}%", "Éves árfolyam-sáv")
    else:
        t4.metric("52 hetes kilengés", "N/A", "Kevés adat")

    st.info(
        f"{eval_res['trend_note']}\n\n"
        f"{eval_res['rsi_note']}\n\n"
        f"{eval_res['fx_note']}\n\n"
        f"{eval_res['matrix_rule_note']}"
    )

    with st.expander("📘 Döntési mátrix puska: Mikor vételi jel az RSI < 30, és hogyan súlyozzuk a devizát?"):
        st.markdown(
            """
- **Emelkedő főtrend (`Ár > SMA 200`) + Alacsony RSI (`< 35`):** Egészséges trendben átmeneti visszaesés → **HÍGÍTÁS (DCA) / ÉRDEMES VENNI**.
- **Emelkedő főtrend (`Ár > SMA 200`) + Semleges RSI (`35–65`):** Egészséges trend, de még nincs diszkont → **Meglévő: TARTÁS**, **Új vétel: KIVÁRÁS**.
- **Megtört főtrend (`Ár < SMA 200`) + Alacsony RSI (`< 32`) – „Hulló kés”:** Eladni a pánik alján már késő (**Meglévő: TARTÁS**), új pénzzel meg kell várni, amíg az RSI visszafordul **35 fölé** (**Új vétel: KIVÁRÁS**).
- **Kategória- és volatilitásfüggő devizasúlyozás:** Egy 10–13%-os USD/HUF vagy EUR/HUF felár **részvénynél, ETF-nél és befektetési jegynél** komoly visszatartó erő, mert az éves hozampotenciál jelentős részét kiteszi. **Kriptovalutáknál** viszont a 100–200%+ éves árfolyam-kilengés mellett a devizafelár **másodlagos súlyú**, ezért erős támasznál / RSI-fordulónál nem akadálya a szakaszos pozícióépítésnek.
- **Tartós eső trend (`Ár < SMA 50 < SMA 200`) + Nem túladott RSI (`≥ 32`):** Kockázatcsökkentés → **RÉSZLEGES KITÁRAZÁS / Új vétel: KERÜLENDŐ**.
            """
        )


def render_asset_details(item, fx_map=None):
    """Megjeleníti egy eszköz grafikonját, a tanuló SMA/RSI értékelőt és a magyar híreket."""
    tech = item["tech"]
    cat_cfg = CATEGORY_CONFIG.get(item.get("category", "STOCKS"), CATEGORY_CONFIG["STOCKS"])
    cat_color = cat_cfg["color"]

    meta_parts = []
    if item.get("source_note"):
        meta_parts.append(f"📡 **Adatforrás:** {item['source_note']}")
    if item.get("strategy"):
        meta_parts.append(f"🧭 **Stratégia:** {item['strategy']}")
    if meta_parts:
        st.caption(" &nbsp;|&nbsp; ".join(meta_parts))

    if tech and tech.get("chart_df") is not None and len(tech["chart_df"]) >= 2:
        st.line_chart(
            tech["chart_df"],
            color=[cat_color, "#A855F7", "#64748B"],
            height=210,
        )
        st.markdown(
            f'<div class="legend-bar">📈 <strong>Grafikon:</strong> '
            f'<span style="color:{cat_color};font-weight:700;">● Árfolyam</span> &nbsp;|&nbsp; '
            f'<span style="color:#64748B;font-weight:700;">● SMA 50 (50 napos átlag)</span> &nbsp;|&nbsp; '
            f'<span style="color:#A855F7;font-weight:700;">● SMA 200 (200 napos átlag)</span></div>',
            unsafe_allow_html=True,
        )

    render_technical_learning_panel(tech, item.get("currency", "USD"), fx_map, item.get("category", "STOCKS"))

    with st.expander("📰 Elemzett hírek (magyarra fordítva)"):
        if item["news_list"]:
            for n in item["news_list"]:
                if n["url"]:
                    st.markdown(f"- [{n['hu_title']}]({n['url']}) *({n['publisher']})*")
                else:
                    st.markdown(f"- {n['hu_title']} *({n['publisher']})*")
        else:
            st.write("Nincsenek friss hírek az elmúlt napokból.")


def run_ui_analysis_with_progress(send_alerts=True):
    """Élő folyamatjelzővel futtatja az elemzést a Streamlit felületen."""
    status_box = st.empty()
    progress_bar = st.progress(0)

    def on_progress(pct, msg):
        progress_bar.progress(min(100, max(0, pct)))
        status_box.info(msg)

    res = run_core_analysis(progress_cb=on_progress, trigger_source="Kézi / UI", send_alerts=send_alerts)
    progress_bar.empty()
    status_box.empty()
    return res


@st.cache_data(ttl=3600, show_spinner=False)
def get_shared_analysis(send_alerts=True):
    """1 órán át szerverszinten tárolja az eredményt, így az oldal újratöltése nem fogyaszt új kvótát."""
    return run_ui_analysis_with_progress(send_alerts=send_alerts)


def render_category_header(cat_key, count):
    """Színkódolt szekciófejlécet jelenít meg mind a listás, mind a kártyás nézetben."""
    cfg = CATEGORY_CONFIG[cat_key]
    color = cfg["color"]
    bg = hex_to_rgba(color, 0.12)
    border = hex_to_rgba(color, 0.35)
    badge_bg = hex_to_rgba(color, 0.20)
    st.markdown(
        f"""
        <div class="category-banner" style="background: {bg}; border: 1px solid {border}; border-left: 6px solid {color};">
            <span style="color: {color};">{cfg['icon']} {cfg['label']}</span>
            <span class="category-count" style="background: {badge_bg}; color: {color};">{count} eszköz</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_streamlit_app():
    """A Streamlit webes felület teljes megjelenítése."""
    st.set_page_config(
        page_title="AI Piaci Iránytű",
        page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🧭",
        layout="wide",
    )

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

        .category-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 16px;
            border-radius: 6px;
            margin: 18px 0 12px 0;
            font-family: 'Exo 2', sans-serif;
            font-weight: 800;
            font-size: 1.12rem;
            letter-spacing: 0.02em;
        }
        .category-count {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.80rem;
            font-weight: 700;
            padding: 2px 10px;
            border-radius: 4px;
        }

        .asset-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }
        .asset-name {
            font-family: 'Exo 2', sans-serif;
            font-size: 1.20rem;
            font-weight: 800;
        }
        .asset-ticker {
            font-family: 'Exo 2', sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
            opacity: 0.75;
            margin-left: 6px;
        }
        .cat-tag {
            display: inline-block;
            font-family: 'Exo 2', sans-serif;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 2px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
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
            border-left: 4px solid #F59E0B;
            background: rgba(245, 158, 11, 0.08);
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
            border-left-color: #F59E0B !important;
            background: rgba(245, 158, 11, 0.12) !important;
            color: #D97706;
        }
        .badge-sell {
            border-left-color: #EF4444 !important;
            background: rgba(239, 68, 68, 0.12) !important;
            color: #DC2626;
        }

        @media (prefers-color-scheme: dark) {
            .badge-buy { color: #34D399; }
            .badge-hold { color: #FBBF24; }
            .badge-sell { color: #F87171; }
        }

        .tech-pill {
            display: inline-block;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 4px;
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
            <p>Gyors portfólió-döntéstámogató • Részvények, Kriptovaluták, ETF-ek, VIG alapok (BAMOSZ) & EUR/USD/HUF hatás</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        icon_b64 = get_icon_base64()
        if icon_b64:
            st.markdown(
                f"""<div style="display: flex; flex-direction: column; align-items: center; text-align: center; padding: 14px 12px 12px 12px; margin-bottom: 16px; background: linear-gradient(135deg, #1E1E24 0%, #2B2D38 100%); border: 1px solid rgba(255, 107, 0, 0.35); border-left: 4px solid #FF6B00; border-radius: 8px;">
<img src="data:image/png;base64,{icon_b64}" style="width: 100%; max-width: 220px; aspect-ratio: 1 / 1; border-radius: 8px; border: 1px solid rgba(255, 107, 0, 0.55); object-fit: cover; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);" />
<div style="margin-top: 11px; font-family: 'Exo 2', sans-serif; font-weight: 800; font-size: 1.05rem; color: #FFFFFF; letter-spacing: 0.04em; line-height: 1.2;">AI PIACI IRÁNYTŰ</div>
<div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.76rem; color: #CBD5E1; margin-top: 3px;">Portfólió &amp; Deviza Radar</div>
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("### ⚡ Vezérlőpult")
        if st.button("🔄 Frissítés most (Élő adat)", use_container_width=True):
            get_shared_analysis.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 🎨 Eszközkategóriák")
        for c_key in CATEGORY_ORDER:
            c_cfg = CATEGORY_CONFIG[c_key]
            st.markdown(
                f'<div style="font-size: 0.84rem; margin-bottom: 4px;">'
                f'<span style="color:{c_cfg["color"]}; font-weight:800;">■</span> {c_cfg["label"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📧 E-mail Riasztások")
        alerts_enabled = st.toggle("Automatikus riasztás", value=True)
        st.caption("0–24h háttérfigyelővel szinkronizálva (**Riasztasok** lapfül, 24h szűrő).")

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
        data, fx_map, last_updated, log_ok, log_msg, ai_err, alert_map = get_shared_analysis(send_alerts=alerts_enabled)
    except Exception as e:
        st.error(f"Hiba az adatok betöltésekor: {e}")
        st.stop()

    if ai_err:
        get_shared_analysis.clear()
        err_u = ai_err.upper()
        if "429" in err_u or "RESOURCE_EXHAUSTED" in err_u:
            st.info(
                "ℹ️ **AI napi/perces kvótalimit elérve:** A portfólió automatikusan a kategória- és "
                "volatilitás-súlyozott SMA 50 / SMA 200 / RSI + EUR/USD devizamátrix alapján jelenik meg."
            )
        elif "503" in err_u or "UNAVAILABLE" in err_u or "TIMEOUT" in err_u:
            st.info(
                "ℹ️ **Az AI szerver átmenetileg túlterhelt:** A portfólió automatikusan a kategória- és "
                "volatilitás-súlyozott SMA 50 / SMA 200 / RSI + EUR/USD devizamátrix alapján jelenik meg."
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
        st.warning("Nincs 'IGEN' státuszú eszköz az 'Eszkozok', 'Kriptovaluták', 'ETF-ek' vagy 'VIG_alapok' munkalapon.")
        st.stop()

    for item in data:
        item_uid = f"{item['category']}_{item['name']}_{item['ticker']}"
        if item_uid in alert_map and alert_map[item_uid][0]:
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
        for cat_key in CATEGORY_ORDER:
            cat_items = [x for x in filtered_data if x.get("category") == cat_key]
            if not cat_items:
                continue

            render_category_header(cat_key, len(cat_items))

            table_rows = []
            for item in cat_items:
                tech = item["tech"]
                price_str = format_price(tech["price"]) if tech else "N/A"
                chg_str = f"{tech['change_pct']:+.2f}%" if tech else "N/A"
                range_str = f"{tech['range_52w_pct']:.0f}%" if (tech and tech.get("range_52w_pct") is not None) else "N/A"

                table_rows.append({
                    "Eszköz": f"{item['name']} ({item['ticker']})",
                    "Deviza": item["currency"],
                    "Árfolyam": price_str,
                    "Napi %": chg_str,
                    "52h sáv": range_str,
                    "Saját pozíció": item["existing_action"],
                    "Új vásárlás": item["new_entry_action"],
                    "Helyzetkép": item["plain_tech"],
                    "Indoklás": item["reasoning"],
                })

            df_list = pd.DataFrame(table_rows)
            table_height = min(520, 42 + len(df_list) * 38)
            st.dataframe(df_list, use_container_width=True, hide_index=True, height=table_height)

        st.markdown("---")
        option_labels = [
            f"[{CATEGORY_CONFIG[item['category']]['short_label']}] {item['name']} ({item['ticker']})"
            for item in filtered_data
        ]
        selected_label = st.selectbox(
            "🔎 Grafikon, SMA/RSI értékelő és magyarított hírek megtekintése ehhez az eszközhöz:",
            option_labels,
        )
        selected_idx = option_labels.index(selected_label)
        selected_item = filtered_data[selected_idx]

        if selected_item:
            sel_cfg = CATEGORY_CONFIG[selected_item["category"]]
            with st.container(border=True):
                st.markdown(
                    f'<div style="border-left: 4px solid {sel_cfg["color"]}; padding-left: 10px; margin-bottom: 8px;">'
                    f'<span style="color:{sel_cfg["color"]}; font-weight:800;">{sel_cfg["icon"]} {sel_cfg["label"]}</span><br>'
                    f'<strong>{selected_item["name"]} ({selected_item["ticker"]})</strong> — '
                    f'Saját pozíció: <strong>{selected_item["existing_action"]}</strong> | '
                    f'Új vétel: <strong>{selected_item["new_entry_action"]}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.write(selected_item["reasoning"])
                render_asset_details(selected_item, fx_map)

    # --- 2. KÁRTYÁS NÉZET ---
    else:
        for cat_key in CATEGORY_ORDER:
            cat_items = [x for x in filtered_data if x.get("category") == cat_key]
            if not cat_items:
                continue

            cfg = CATEGORY_CONFIG[cat_key]
            cat_color = cfg["color"]
            cat_bg = hex_to_rgba(cat_color, 0.12)
            cat_border = hex_to_rgba(cat_color, 0.35)

            render_category_header(cat_key, len(cat_items))

            cols = st.columns(2)
            for idx, item in enumerate(cat_items):
                item_uid = f"{item['category']}_{item['name']}_{item['ticker']}"
                alert_status = alert_map.get(item_uid)
                with cols[idx % 2]:
                    with st.container(border=True):
                        tech = item["tech"]

                        st.markdown(
                            f'<div style="height: 4px; background: {cat_color}; border-radius: 3px; margin: -4px 0 10px 0;"></div>'
                            f'<div class="cat-tag" style="background: {cat_bg}; color: {cat_color}; border: 1px solid {cat_border};">'
                            f'{cfg["icon"]} {cfg["short_label"]}</div>',
                            unsafe_allow_html=True,
                        )

                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown(
                                f'<div class="asset-header" style="border-bottom: 2px solid {cat_border};">'
                                f'<div><span class="asset-name" style="color: {cat_color};">{item["name"]}</span>'
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
                            st.markdown(
                                f'<div class="tech-pill" style="background: {cat_bg}; border: 1px solid {cat_border};">'
                                f'🧭 {item["plain_tech"]}</div>',
                                unsafe_allow_html=True,
                            )

                        st.write(item["reasoning"])

                        if alert_status:
                            _, msg_text = alert_status
                            st.caption(f"🔔 Riasztás: **{msg_text}**")

                        render_asset_details(item, fx_map)


if __name__ == "__main__":
    render_streamlit_app()
