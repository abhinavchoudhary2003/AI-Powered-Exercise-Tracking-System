from models.database import init_db, create_user, log_exercise, get_daily_totals
from datetime import date

init_db()
uid = create_user("Test User", 25, 175, 70)
log_exercise(uid, "pushup", 12)
log_exercise(uid, "squat", 20)
log_exercise(uid, "pushup", 8)
print(get_daily_totals(uid, date.today().isoformat()))
# Expected: {'pushup': 20, 'squat': 20}