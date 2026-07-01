"""API 服务测试"""
import uvicorn
import threading
import time
import requests
import json

# 启动服务
def run_server():
    from api import app
    uvicorn.run(app, host='127.0.0.1', port=8765, log_level='warning')

print("Starting API server...")
t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(2)

# 测试 1: Docs
r = requests.get('http://127.0.0.1:8765/docs', timeout=5)
print(f'[1] Docs: {r.status_code}')

# 测试 2: Chat
payload = {'message': '列出所有学生', 'db_id': 'student_db'}
r = requests.post('http://127.0.0.1:8765/api/v1/chat', json=payload, timeout=30)
print(f'[2] Chat: {r.status_code}')
data = r.json()
print(f'  SQL: {data.get("sql", "?")}')
print(f'  Success: {data.get("success", "?")}')
print(f'  Reply: {data.get("reply", "?")}')
result = data.get('result', {})
if result:
    print(f'  Rows: {result.get("row_count", 0)}')

session_id = data.get('session_id', '')

# 测试 3: Follow-up
payload2 = {'session_id': session_id, 'message': '只看计算机科学专业的'}
r2 = requests.post('http://127.0.0.1:8765/api/v1/chat', json=payload2, timeout=30)
print(f'[3] Follow-up: {r2.status_code}')
d2 = r2.json()
print(f'  SQL: {d2.get("sql", "?")}')
print(f'  Success: {d2.get("success", "?")}')

# 测试 4: History
r3 = requests.get(f'http://127.0.0.1:8765/api/v1/session/{session_id}/history', timeout=10)
print(f'[4] History: {r3.status_code}')
d3 = r3.json()
print(f'  Turns: {len(d3.get("turns", []))}')

# 测试 5: Delete session
r4 = requests.delete(f'http://127.0.0.1:8765/api/v1/session/{session_id}', timeout=10)
print(f'[5] Delete: {r4.status_code} {r4.json()}')

print("\nAll API tests completed!")
