import sys
from market_analyzer import run_core_analysis


def main():
    print("🚀 AI Piaci Iránytű – Óránkénti 0–24h háttérfuttatás indítása...")
    results, fx_map, run_ts, log_ok, log_msg, ai_err, alert_map = run_core_analysis(
        progress_cb=lambda pct, msg: print(f"[{pct:3d}%] {msg}"),
        trigger_source="Automata 0-24h",
        send_alerts=True,
    )

    print(f"⏱️ Időpont: {run_ts}")
    print(f"📊 Feldolgozott eszközök: {len(results)} db")
    print(f"💾 Naplózás: {log_msg}")

    if ai_err:
        print(f"⚠️ AI megjegyzés (szabályalapú tartalék futott): {ai_err}")

    sent_count = sum(1 for v in alert_map.values() if v[0])
    print(f"📧 Újonnan elküldött e-mail riasztások: {sent_count} db")

    if not log_ok and len(results) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
