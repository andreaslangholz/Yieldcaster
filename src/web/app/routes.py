from app import app
from flask import render_template

@app.route('/')
def index():
    user = {
        'name':'Chandra'
    }
    return render_template('index.html', user=user)