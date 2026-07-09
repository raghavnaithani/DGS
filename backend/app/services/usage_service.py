from fastapi import HTTPException
from datetime import datetime, timezone
from typing import Optional
from contextlib import closing
from app.database.connection import get_connection
from app.config import settings

def check_and_increment_graph_counter(db_path: str, user_id: str) -> None:
    # Open connection
    with closing(get_connection(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT graphs_this_month, subscription_tier, month_reset_at FROM user_profiles WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return  # allow through
            
        graphs_this_month = row["graphs_this_month"]
        subscription_tier = row["subscription_tier"]
        month_reset_at = row["month_reset_at"]
        
        if subscription_tier == 'pro':
            cursor.execute(
                "UPDATE user_profiles SET graphs_this_month = graphs_this_month + 1 WHERE id = ?",
                (user_id,)
            )
            conn.commit()
            return
            
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        stored_month = month_reset_at[:7] if month_reset_at else ""
        
        if current_month != stored_month:
            graphs_this_month = 0
            new_reset_at = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
            cursor.execute(
                "UPDATE user_profiles SET graphs_this_month = 0, month_reset_at = ? WHERE id = ?",
                (new_reset_at, user_id)
            )
            conn.commit()
            
        if graphs_this_month >= settings.free_tier_graph_limit:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "Payment Required",
                    "message": "You have reached your free tier graph limit for this month.",
                    "upgrade_url": "/dashboard"
                }
            )
            
        cursor.execute(
            "UPDATE user_profiles SET graphs_this_month = graphs_this_month + 1 WHERE id = ?",
            (user_id,)
        )
        conn.commit()
