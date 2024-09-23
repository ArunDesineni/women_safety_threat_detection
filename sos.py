from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse
from jinja2 import Template

app = FastAPI()

# Store alerts
alerts = []

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Emergency Alerts</title>
</head>
<body>
    <h1>Emergency Alerts</h1>
    <form action="/send-alert/" method="post">
        <input type="submit" value="Send SOS Alert">
    </form>
    <ul>
        {% for alert in alerts %}
        <li>Alert from IP: {{ alert }}</li>
        {% endfor %}
    </ul>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_alerts():
    template = Template(html_template)
    return template.render(alerts=alerts)

@app.post("/send-alert/")
async def send_alert(request: Request):
    
    client_ip = request.client.host
    alerts.append(client_ip)
    return RedirectResponse(url="/", status_code=302)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="172.22.7.128", port=8000)
