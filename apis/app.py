import math
from flask import Flask, redirect, url_for,request,render_template,jsonify
import pickle
import numpy as np
import sys
from flask_cors import CORS, cross_origin



sys.path.insert(0,'C:/Users/ankit/OneDrive/Desktop/ml-model/models')
from model import prediction
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# @app.route('/')
# def hello():
#     return 'hello world'

# app.add_url_rule('/','',hello)

# @app.route('/hello/<name>')
# def hello_name(name):
#     return 'my name is %s' % name

# @app.route('/marks/<marks>')
# def marks(marks):
#     return "my marks is %s" %marks

# @app.route('/send/<age>')
# def send(age):
#     if int(age)>18:
#         return redirect(url_for('marks',marks=70)) 
#     else:
#         return redirect(url_for('hello'))
# @app.route('/forms',methods=['POST','GET'])
# def forms():
#     if request.method == 'POST':
#         user = request.form['names']
#         number = request.form['num']
#         return "my name is %s and my number is %s" % (user, number)
#     else :
#         return redirect(url_for('hello_name'))

# @app.route('/rendering/<int:score>',methods=['POST','GET'])
# def rendering(score):
#     return render_template('login.html',marks=score)

model = pickle.load(open('model.pkl','rb'))

@app.route('/api',methods = ['POST'])
@cross_origin()
def api():
    data = request.get_json(force=True)
    predict = prediction(np.array(data['exp']))
    output = predict.tolist()
    
    return {"value":output[0][0]}

if __name__ == ("__main__") :
    app.run(debug=True)
   
   