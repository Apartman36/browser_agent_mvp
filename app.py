from flask import Flask, render_template, request, jsonify
import threading
import sqlite3

from agent.core import run_agent, interrupt_event, user_input_queue, ui_logs
from agent.browser import Browser
from agent.logging_utils import read_action_log

app = Flask(__name__)

agent_thread = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logs")
def logs():
    start_idx = int(request.args.get("start", 0))
    new_logs = ui_logs[start_idx:]
    return jsonify({"logs": new_logs, "next_index": len(ui_logs)})

@app.route("/state")
def get_state():
    # Read memory facts from db
    facts = {}
    try:
        with sqlite3.connect("memory.db") as conn:
            cursor = conn.execute("SELECT key, value FROM facts")
            for row in cursor:
                facts[row[0]] = row[1]
    except Exception:
        pass

    # Get history length roughly from action logs
    history_len = 0
    try:
        history_len = len(read_action_log()[-8:])
    except Exception:
        pass

    return jsonify({
        "facts": facts,
        "history_len": history_len,
        "is_running": agent_thread is not None and agent_thread.is_alive()
    })

@app.route("/input", methods=["POST"])
def handle_input():
    global agent_thread
    text = request.json.get("text")

    # If thread is not alive, start a new agent session
    if agent_thread is None or not agent_thread.is_alive():
        interrupt_event.clear()
        ui_logs.clear()
        ui_logs.append(f"Starting task: {text}")

        def run():
            with Browser() as browser:
                run_agent(goal=text, browser=browser, max_steps=15)

        agent_thread = threading.Thread(target=run)
        agent_thread.start()
        return jsonify({"status": "started"})
    else:
        # Agent is running, feed it to the queue
        user_input_queue.append(text)
        return jsonify({"status": "input_queued"})

@app.route("/interrupt", methods=["POST"])
def interrupt():
    interrupt_event.set()
    return jsonify({"status": "interrupt_sent"})

@app.route("/clear", methods=["POST"])
def clear():
    global agent_thread
    if agent_thread is not None and agent_thread.is_alive():
        return jsonify({"status": "error", "message": "Cannot clear while agent is running"}), 400

    ui_logs.clear()
    user_input_queue.clear()
    interrupt_event.clear()

    # Clear memory DB
    try:
        with sqlite3.connect("memory.db") as conn:
            conn.execute("DELETE FROM facts")
    except Exception:
        pass

    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
