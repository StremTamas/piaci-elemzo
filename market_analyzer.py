import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import json
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timedelta

# Oldal beállítása
st.set_page_config(page_title="AI Piaci & Technikai Elemző", page_icon="📈", layout="wide")

# Gemini API inicializálása a Streamlit Secrets-ből
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_gspread_client():
    """Hitelesített Google Sheets kliens létrehozása írási és olvasási jogosultsággal."""
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_resource
def get_alert_history():
    """Szerverszintű memória a már elküldött riasztások nyilvántartására (duplikáció ellen)."""
    return {}


def send_gmail_alert(subject, plain_body, html_body):
    """E-mail küldése Gmail SMTP-n keresztül Alkalmazásjelszó (App Password) használatával."""
    sender_email = st.secrets["SMTP_EMAIL"]
    app_password = st.secrets["SMTP_APP_PASSWORD"]
    recipient_email = st.secrets.get("ALERT_RECIPIENT_EMAIL", sender_email)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"AI Piaci Riasztó <{sender_email}>"
    msg["To"] = recipient_email
    msg.set_content(plain_body)
    msg.add_alternative(html_body, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)


def check_and_send_alert(item):
    """Ellenőrzi a riasztási feltételeket (ELADÁS vagy RSI < 30) és 24 óránként max. 1 levelet küld."""
    rec = item.get("recommendation", "")
    tech = item.get("tech") or {}
    rsi = tech.get("rsi_14")

    triggers = []
    if rec == "ELADÁS":
        triggers.append("AI ELADÁS jelzés")
    if rsi is not None and rsi < 30:
        triggers.append(f"Túladott RSI ({rsi:.1f} < 30)")

    if not triggers:
        return False, None

    alert_key = f"{item['ticker']}_{'_'.join(triggers)}"
    history = get_alert_history()
    now = datetime.now()

    if alert_key in history and (now - history[alert_key]) < timedelta(hours=24):
        return False, "Már riasztva (24h)"

    trigger_text = " + ".join(triggers)
    price_str = f"{tech['price']:,.2f}" if tech.get("price") is not None else "N/A"
    change_str = f"{tech['change_pct']:+.2f}%" if tech.get("change_pct") is not None else "N/A"
    rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
    sma50_str = f"{tech['sma_50']:,.2f}" if tech.get("sma_50") is not None else "N/A"
    sma200_str = f"{tech['sma_200']:,.2f}" if tech.get("sma_200") is not None else "N/A"

    subject = f"🚨 Piaci Riasztás: {item['name']} ({item['ticker']}) – {trigger_text}"

    plain_body = (
        f"Piaci riasztás érkezett az alábbi eszközre:\n\n"
        f"Eszköz: {item['name']} ({item['ticker']})\n"
        f"Kiváltó ok: {trigger_text}\n"
        f"Aktuális ár: {price_str} ({change_str})\n"
        f"Ajánlás: {rec} | Hangulat pontszám: {item['score']}/100\n"
        f"RSI (14): {rsi_str} | SMA 50: {sma50_str} | SMA 200: {sma200_str}\n\n"
        f"AI Indoklás:\n{item['reasoning']}\n"
    )

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #222; line-height: 1.5;">
        <h2 style="color: #c0392b; margin-bottom: 4px;">🚨 Piaci Riasztás: {item['name']} ({item['ticker']})</h2>
        <p style="margin-top: 0; font-size: 15px;"><strong>Kiváltó ok:</strong> {trigger_text}</p>
        <table style="border-collapse: collapse; width: 100%; max-width: 520px; margin: 12px 0;">
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; background: #f8f9fa;"><strong>Aktuális ár</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{price_str} ({change_str})</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; background: #f8f9fa;"><strong>AI Ajánlás & Pontszám</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>{rec}</strong> ({item['score']}/100)</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; background: #f8f9fa;"><strong>RSI (14)</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{rsi_str}</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; background: #f8f9fa;"><strong>SMA 50 / SMA 200</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{sma50_str} / {sma200_str}</td>
          </tr>
        </table>
        <p><strong>AI Elemzés és indoklás:</strong><br>{item['reasoning']}</p>
        <hr style="border: none; border-top: 1px solid #eee; margin-top: 20px;">
        <p style="font-size: 12px; color: #777;">Automatikus értesítés a Streamlit AI Portfólió Elemző rendszerből.</p>
      </body>
    </html>
    """

    try:
        send_gmail_alert(subject, plain_body, html_body)
        history[alert_key] = now
        return True, "Riasztás elküldve"
    except Exception as e:
        return False, f"SMTP hiba: {e}"


def get_active_assets_from_sheet(client):
    """Beolvassa az 'IGEN' státuszú eszközöket az 'Eszkozok' munkalapról."""
    sheet = client.open_by_key(st.secrets["SHEET_ID"]).worksheet("Eszkozok")
    records = sheet.get_all_records()

    return [
        row for row in records
        if str(row.get("aktiv", "")).strip().upper() == "IGEN" and row.get("ticker")
    ]


def log_results_to_sheet(client, results, run_timestamp):
    """Kötegelve menti az elemzési eredményeket a Google Táblázat 'Naplo' munkalapjára."""
    if not results:
        return True, "Nincs menthető adat."

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
            item["recommendation"],
            item["technical_state"],
            item["risk"],
            item["reasoning"],
        ])

    naplo_sheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
    return True, f"Naplózva a 'Naplo' munkalapra ({len(rows_to_append)} sor)."


def calculate_technical_indicators(hist_df):
    """Kiszámolja a 14 napos RSI-t, az 50/200 napos mozgóátlagot és előkészíti a grafikon adatokat."""
    if hist_df is None or hist_df.empty or "Close" not in hist_df:
        return None

    close = hist_df["Close"].dropna()
    if len(close) < 2:
        return None

    current_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    change_pct = ((current_price - prev_close) / prev_close) * 100

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
            trend = "Erős emelkedő (Ár > SMA50 > SMA200)"
        elif current_price < sma_50 < sma_200:
            trend = "Erős csökkenő (Ár < SMA50 < SMA200)"
        elif sma_50 > sma_200:
            trend = "Emelkedő trend, rövid távú korrekcióval"
        else:
            trend = "Csökkenő trend, rövid távú felpattanással"
    else:
        trend = "Nincs elegendő historikus adat a 200 napos trendhez"

    # Utolsó ~1 év (252 nap) előkészítése az interaktív grafikonhoz
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


def fetch_yahoo_data(ticker_symbol):
    """Lekéri a 2 éves grafikonadatokat, indikátorokat és a friss híreket."""
    ticker = yf.Ticker(ticker_symbol)

    try:
        hist = ticker.history(period="2y", interval="1d")
        tech_data = calculate_technical_indicators(hist)
    except Exception:
        tech_data = None

    news_summaries = []
    try:
        raw_news = ticker.news[:6] if ticker.news else []
        for item in raw_news:
            content = item.get("content", item)
            title = content.get("title", "")
            provider = content.get("provider", {})
            publisher = provider.get("displayName") if isinstance(provider, dict) else item.get("publisher", "Ismeretlen")
            if title:
                news_summaries.append(f"- {title} (Forrás: {publisher})")
    except Exception:
        pass

    news_text = "\n".join(news_summaries) if news_summaries else "Nincsenek friss hírek az elmúlt napokból."
    return tech_data, news_text


def analyze_asset_with_ai(name, ticker, tech_data, news_text):
    """Kombinált híralapú és technikai elemzés Gemini AI segítségével."""
    if tech_data:
        rsi_str = f"{tech_data['rsi_14']:.1f}" if tech_data["rsi_14"] is not None else "N/A"
        sma50_str = f"{tech_data['sma_50']:,.2f}" if tech_data["sma_50"] is not None else "N/A"
        sma200_str = f"{tech_data['sma_200']:,.2f}" if tech_data["sma_200"] is not None else "N/A"
        tech_summary = (
            f"Aktuális ár: {tech_data['price']:,.2f} (Napi változás: {tech_data['change_pct']:+.2f}%)\n"
            f"14 napos RSI: {rsi_str} (30 alatt túladott, 70 felett túlvett)\n"
            f"50 napos mozgóátlag (SMA 50): {sma50_str}\n"
            f"200 napos mozgóátlag (SMA 200): {sma200_str}\n"
            f"Technikai trendkép: {tech_data['trend']}"
        )
    else:
        tech_summary = "Technikai adatok nem elérhetők."

    prompt = f"""
    Szenior pénzügyi és technikai piaci elemzőként viselkedsz.
    Eszköz: {name} ({ticker})

    --- TECHNIKAI INDIKÁTOROK (GRAFIKON ADATOK) ---
    {tech_summary}

    --- LEGFREISSEBB PIACI HÍREK ---
    {news_text}

    Feladat: Végezz kombinált elemzést! Vesd össze a hírek hangulatát a technikai indikátorokkal (például ha jók a hírek, de az RSI 70 felett van, jelezd a túlvettséget; ha pánik van, de az RSI 30 alatt van és az SMA 200 tartja az árat, jelezd a technikai támaszt).
    Válaszolj KIZÁRÓLAG érvényes JSON formátumban, az alábbi kulcsokkal:
    - "sentiment_score": egész szám 1 és 100 között (1 = extrém eladási nyomás, 50 = semleges, 100 = erős vételi jelzés)
    - "recommendation": pontosan az egyik érték ezek közül: "VÉTEL", "TARTÁS", vagy "ELADÁS"
    - "technical_state": rövid 2-3 szavas technikai címke (pl. "BIKA / TÚLVETT", "MEDVE / TÚLADOTT", "SEMLEGES TREND")
    - "risk_level": "ALACSONY", "KÖZEPES" vagy "MAGAS"
    - "reasoning": tömör, 3 mondatos magyar indoklás, amely egyszerre hivatkozik a konkrét hírekre ÉS az RSI / mozgóátlag értékekre.
    """
    try:
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        return {
            "sentiment_score": 50,
            "recommendation": "HIBA",
            "technical_state": "ISMERETLEN",
            "risk_level": "ISMERETLEN",
            "reasoning": f"Nem sikerült az AI elemzés: {str(e)}",
        }


@st.cache_data(ttl=3600, show_spinner=False)
def run_full_analysis():
    """Futtatja a teljes elemzést, naplózza az eredményt és 1 óráig gyorsítótárazza."""
    client = get_gspread_client()
    assets = get_active_assets_from_sheet(client)
    results = []

    for asset in assets:
        ticker = str(asset["ticker"]).strip()
        name = str(asset.get("name", ticker)).strip()

        tech_data, news_text = fetch_yahoo_data(ticker)
        ai_res = analyze_asset_with_ai(name, ticker, tech_data, news_text)

        results.append({
            "name": name,
            "ticker": ticker,
            "tech": tech_data,
            "score": ai_res.get("sentiment_score", 50),
            "recommendation": ai_res.get("recommendation", "TARTÁS"),
            "technical_state": ai_res.get("technical_state", ""),
            "risk": ai_res.get("risk_level", "KÖZEPES"),
            "reasoning": ai_res.get("reasoning", ""),
            "news": news_text,
        })

    run_timestamp = datetime.now().strftime("%Y.%m.%d. %H:%M")

    try:
        log_ok, log_msg = log_results_to_sheet(client, results, run_timestamp)
    except Exception as e:
        log_ok, log_msg = False, f"Naplózási hiba (ellenőrizd a Szerkesztő jogosultságot): {e}"

    return results, run_timestamp, log_ok, log_msg


# --- FELÜLET (UI) ---
st.title("📊 AI Portfólió: Hírek, Technika & E-mail Riasztás")

with st.sidebar:
    st.header("Vezérlőpult")
    if st.button("🔄 Frissítés most (Élő adat)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("📧 Gmail Riasztások")
    alerts_enabled = st.toggle("Automatikus e-mail riasztás", value=True)
    st.caption("Feltétel: **ELADÁS** ajánlás vagy **30 alatti RSI** érték.")

    if st.button("Teszt e-mail küldése", use_container_width=True):
        try:
            send_gmail_alert(
                "✅ Teszt riasztás – AI Piaci Elemző",
                "A Gmail SMTP kapcsolat megfelelően működik a Streamlit felhőből.",
                "<p>A <strong>Gmail SMTP</strong> kapcsolat megfelelően működik a Streamlit felhőből.</p>",
            )
            st.success("Teszt e-mail sikeresen elküldve!")
        except Exception as e:
            st.error(f"Hiba a küldéskor: {e}")

    st.markdown("---")
    st.markdown(
        "**Technikai jelmagyarázat:**\n"
        "- **RSI 14:** 30 alatt túladott, 70 felett túlvett.\n"
        "- **SMA 50 / 200:** Közép- és hosszú távú trendvonalak."
    )

with st.spinner("Grafikonadatok, indikátorok és hírek elemzése..."):
    try:
        data, last_updated, log_ok, log_msg = run_full_analysis()
    except Exception as e:
        st.error(f"Hiba a Google Táblázat elérésekor: {e}")
        st.stop()

st.caption(f"Utolsó frissítés ideje: **{last_updated}**")
if log_ok:
    st.caption(f"💾 {log_msg}")
else:
    st.warning(f"⚠️ {log_msg}")

if not data:
    st.warning("Nincs 'IGEN' státuszú eszköz a Google Táblázatban.")
    st.stop()

cols = st.columns(2)
for idx, item in enumerate(data):
    alert_status = None
    if alerts_enabled:
        sent, status_msg = check_and_send_alert(item)
        if status_msg:
            alert_status = (sent, status_msg)

    with cols[idx % 2]:
        with st.container(border=True):
            tech = item["tech"]
            c1, c2 = st.columns([2, 1])
            c1.subheader(f"{item['name']} ({item['ticker']})")

            if tech:
                c2.metric("Árfolyam", f"{tech['price']:,.2f}", f"{tech['change_pct']:+.2f}%")
            else:
                c2.metric("Árfolyam", "N/A")

            if tech:
                t1, t2, t3 = st.columns(3)
                rsi_val = f"{tech['rsi_14']:.1f}" if tech["rsi_14"] is not None else "N/A"
                sma50_val = f"{tech['sma_50']:,.2f}" if tech["sma_50"] is not None else "N/A"
                sma200_val = f"{tech['sma_200']:,.2f}" if tech["sma_200"] is not None else "N/A"

                rsi_delta = "Túlvett" if (tech["rsi_14"] and tech["rsi_14"] > 70) else ("Túladott" if (tech["rsi_14"] and tech["rsi_14"] < 30) else "Semleges")
                t1.metric("RSI (14)", rsi_val, rsi_delta, delta_color="off")
                t2.metric("SMA 50", sma50_val)
                t3.metric("SMA 200", sma200_val)

                # Interaktív árfolyam- és mozgóátlag-grafikon
                chart_df = tech.get("chart_df")
                if chart_df is not None and not chart_df.empty:
                    st.line_chart(
                        chart_df,
                        color=["#2980b9", "#8e44ad", "#f39c12"],
                        height=230,
                    )

            rec = item["recommendation"]
            badge_color = "green" if rec == "VÉTEL" else ("red" if rec == "ELADÁS" else "orange")
            st.markdown(
                f"**Ajánlás:** :{badge_color}[**{rec}**] &nbsp;|&nbsp; "
                f"**Pontszám:** {item['score']}/100 &nbsp;|&nbsp; "
                f"**Technikai kép:** {item['technical_state']}"
            )

            if alert_status:
                is_sent, msg_text = alert_status
                if is_sent:
                    st.toast(f"Riasztás elküldve: {item['name']} ({msg_text})", icon="📧")
                st.caption(f"🔔 Riasztási állapot: **{msg_text}**")

            st.progress(max(0, min(100, int(item["score"]))))
            st.write(item["reasoning"])

            with st.expander("Elemzett Yahoo Finance hírek megtekintése"):
                st.text(item["news"])
