import sqlite3
c = sqlite3.connect("backend/data/prescription_feedback.db")
c.execute("DELETE FROM feedback"); c.execute("DELETE FROM outcomes"); c.commit(); c.close()
print("feedback e outcomes limpos")