import os, json
import datetime, argparse
from functools import cmp_to_key
from flask import Flask, render_template, request

parser = argparse.ArgumentParser()
# Add argument
parser.add_argument('--host', help='service ip address, default: 127.0.0.1', default='127.0.0.1')
parser.add_argument('-p', '--port', help='service port, default: 8089', default='8089')
parser.add_argument('-l', '--logdir', help='models directory', default='.')

app = Flask(__name__)
app.debug = True

logdir = os.path.abspath('.')

@app.route('/')
def index():
    global logdir

    model_list = []
    for folder in os.listdir(logdir):
        model_list.append(folder)

    # sort model list by time
    model_list.sort()
    return render_template('index.html', model_list = model_list)

@app.route('/notes', methods=['POST'])
def notes():
    global logdir
    
    model_name = request.form.get('model_name')
    notes_path = logdir + '/' + model_name + '/notes.json'

    with open(notes_path) as file:
        notes = json.load(file)

    return json.dumps(notes)

def main():
    global logdir

    # Parse argument
    args = parser.parse_args()
    logdir = args.logdir

    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()