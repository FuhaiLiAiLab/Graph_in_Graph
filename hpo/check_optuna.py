import optuna
import sqlite3
import os

db_path = 'optuna_gigtransformer_copy2.db'
if os.path.exists(db_path):
    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trials'")
    if cursor.fetchone():
        # 获取试验数量
        cursor.execute("SELECT COUNT(*) FROM trials")
        trial_count = cursor.fetchone()[0]
        
        # 获取完成的试验数量
        cursor.execute("SELECT COUNT(*) FROM trials WHERE state = 1")  # 1 = COMPLETE
        complete_count = cursor.fetchone()[0]
        
        print(f'总试验数: {trial_count}')
        print(f'已完成试验: {complete_count}')
        
        # 获取最佳试验
        cursor.execute("""
            SELECT trial_id, value 
            FROM trial_values 
            ORDER BY value DESC 
            LIMIT 1
        """)
        best_result = cursor.fetchone()
        if best_result:
            print(f'最佳试验ID: {best_result[0]}, 最佳值: {best_result[1]:.4f}')
            
        # 获取最近5个试验的状态
        cursor.execute("""
            SELECT trial_id, state, datetime(datetime_complete, 'unixepoch') as complete_time
            FROM trials 
            ORDER BY trial_id DESC 
            LIMIT 5
        """)
        print("\n最近5个试验:")
        for row in cursor.fetchall():
            state_map = {0: 'RUNNING', 1: 'COMPLETE', 2: 'PRUNED', 3: 'FAIL'}
            print(f"  试验 {row[0]}: 状态={state_map.get(row[1], 'UNKNOWN')}, 完成时间={row[2]}")
    else:
        print('trials表不存在')
    conn.close()
else:
    print('数据库文件不存在')