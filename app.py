from flask import Flask, render_template_string, request, jsonify
import threading

from agent.core import run_agent, interrupt_event, user_input_queue, ui_logs
from agent.browser import Browser

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Web UI</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #chat { background: #f9f9f9; padding: 15px; height: 400px; overflow-y: scroll; border: 1px solid #ddd; margin-bottom: 20px; }
        .controls { display: flex; gap: 10px; }
        input[type="text"] { flex-grow: 1; padding: 8px; }
        button { padding: 8px 16px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Autonomous Browser Agent</h1>
    <div id="chat"></div>
    <div class="controls">
        <input type="text" id="userInput" placeholder="Type a task or answer...">
        <button onclick="sendInput()">Send / Start</button>
        <button onclick="interruptAgent()">Interrupt</button>
    </div>

    <script>
        let lastLogIndex = 0;

        function pollLogs() {
            fetch('/logs?start=' + lastLogIndex)
            .then(r => r.json())
            .then(data => {
                const chat = document.getElementById("chat");
                if (data.logs && data.logs.length > 0) {
                    data.logs.forEach(log => {
                        const div = document.createElement("div");
                        div.innerText = log;
                        chat.appendChild(div);
                    });
                    lastLogIndex = data.next_index;
                    chat.scrollTop = chat.scrollHeight;
                }
            });
        }

        setInterval(pollLogs, 1000);

        function sendInput() {
            const text = document.getElementById("userInput").value;
            if (!text) return;
            document.getElementById("userInput").value = "";
            fetch('/input', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
        }

        function interruptAgent() {
            fetch('/interrupt', {method: 'POST'});
        }
    </script>
</body>
</html>
"""

agent_thread = None

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/logs")
def logs():
    start_idx = int(request.args.get("start", 0))
    new_logs = ui_logs[start_idx:]
    return jsonify({"logs": new_logs, "next_index": len(ui_logs)})

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

if __name__ == "__main__":
    app.run(port=5000, debug=True)
