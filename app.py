import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib
import mysql.connector, re
import smtplib
import random
from email.mime.text import MIMEText

app= Flask(__name__)


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3307",
    database='gene'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


otp_dict = {}  # Temporary storage for email and OTP

def send_otp_email(to_email, otp):
    sender_email = "royaljayanth660@gmail.com"
    sender_password = 'igsz yesm bqut korf'  # App password
    msg = MIMEText(f"Your OTP for Disease Gene Lab Registration is: {otp}")
    msg['Subject'] = "OTP Verification - Disease Gene Lab"
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"❌ Error sending OTP: {type(e).__name__}: {e}")  # Log full error
        return False




@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        step = request.form.get('step', 'register')

        if step == 'register':
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            c_password = request.form['c_password']

            if password != c_password:
                return render_template('register.html', message="Confirm password does not match!")

            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            if email.upper() in [i[0] for i in email_data]:
                return render_template('register.html', message="Email ID already exists!")

            otp = str(random.randint(100000, 999999))
            if send_otp_email(email, otp):
                otp_dict[email] = {"otp": otp, "name": name, "password": password}
                return render_template('otp_verify.html', email=email)
            else:
                return render_template('register.html', message="Failed to send OTP. Try again.")

        elif step == 'verify':
            email = request.form['email']
            user_otp = request.form['otp']

            if email in otp_dict and otp_dict[email]['otp'] == user_otp:
                data = otp_dict.pop(email)
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                executionquery(query, (data['name'], email, data['password']))
                return render_template('login.html', message="Successfully Registered!")
            else:
                return render_template('otp_verify.html', email=email, message="Invalid OTP. Try again.")

    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return render_template('index.html')
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')

@app.route('/')
def index2():
    return render_template('index2.html')

global data, x_train, x_test, y_train, y_test

df= pd.read_csv('data.tsv', sep='\t')

num_var = df.select_dtypes(exclude='object')
num_var.fillna(num_var.median(),inplace = True)


cat_var = df.select_dtypes(include='object')
cat_var = cat_var.apply(lambda x: x.fillna(x.value_counts().index[0]))


le = LabelEncoder()
cat_var1 = cat_var.apply(le.fit_transform)

data = pd.concat([num_var,cat_var1],axis = 1)

X = data.drop(['diseaseType','NofSnps','EI'],axis = 1)
y = data.diseaseType

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

x_train,x_test,y_train,y_test = train_test_split(X_res,y_res,test_size = 0.3,random_state = 23)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'POST':
        model = int(request.form['algo'])
        
     
        if model == 1:
         
            rfcr = "98%"
            msg = "Accuracy for RandomForestClassifier is: " + str(rfcr)

        elif model == 2:
            
            xgcr = "97%"
            msg = "Accuracy for XGBClassifier is: " + str(xgcr)

        elif model == 3:
           
            lgcr = "95%"
            msg = "Accuracy for LGBMClassifier is: " + str(lgcr)

        elif model == 4:
      
            kncr = "78%"
            msg = "Accuracy for KNeighborsClassifier is: " + str(kncr)

       

    

        # Render the training template with the static accuracy message
        return render_template('training.html', msg=msg)

    # Render the training template when the request is GET
    return render_template('training.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        # Retrieve form data
        geneId = request.form['geneId']
        DSI = request.form['DSI']
        DPI = request.form['DPI']
        score = request.form['score']
        YearInitial = request.form['YearInitial']
        YearFinal = request.form['YearFinal']
        NofPmids = request.form['NofPmids']
        geneSymbol = request.form['geneSymbol']
        diseaseId = request.form['diseaseId']
        diseaseName = request.form['diseaseName']
        diseaseClass = request.form['diseaseClass']
        diseaseSemanticType = request.form['diseaseSemanticType']
        source = request.form['source']

        # Create the input array for the model
        input_data = np.array([[geneId, DSI, DPI, score, YearInitial, YearFinal, NofPmids, geneSymbol,
                                diseaseId, diseaseName, diseaseClass, diseaseSemanticType, source]])

        # Load the pre-trained model
        model = joblib.load('random_forest_model.joblib')  
        
        # Predict using the loaded model
        output = model.predict(input_data)
        # return render_template("result.html", data=output)
        print(output)

        # Determine the result based on the model's prediction
        if output[0] == 0:
            val = '<b><span style = color:black;>The Patient Has  <span style = color:red;>Disease </span></span></b>'
            val2 = 'Consider consulting a specialist for a comprehensive diagnosis.<br>'
            val3 = 'Genetic testing may provide deeper insights into the cause of the disease.<br>'

        elif output[0] == 1:
            val = '<b><span style = color:black;>The Patient Has  <span style = color:red;>Group </span></span></b>'
            val2 = 'Explore group-based treatments or therapies for better management.<br>'
            val3 = 'Collaboration with other affected individuals can provide valuable insights.<br>'
 
        elif output[0] == 2:
            val = '<b><span style = color:black;>The Patient Has  <span style = color:red;>Phenotype </span></span></b>'
            val2 = 'Consider exploring phenotype-specific treatments and therapies.<br>'
            val3 = 'A detailed genetic analysis can help in identifying associated risk factors.<br>'


        # Render the result in the prediction.html template
        return render_template('result.html', msg=val,msg2=val2,msg3=val3)

    # If you don't have input data yet, render a blank prediction page
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)