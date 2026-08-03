import sqlite3, statistics
con = sqlite3.connect("backend/data/prescription_feedback.db")
t = sorted(r[0] for r in con.execute(
    "SELECT response_time_ms FROM analyses WHERE response_time_ms IS NOT NULL"))
print(f"n={len(t)}  media={statistics.mean(t):.1f} ms  p95={t[int(0.95*len(t))-1]:.1f} ms  max={t[-1]:.1f} ms")