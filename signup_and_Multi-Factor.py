# We are importing the require library
from flask import Flask, render_template, request, redirect, url_for, session
from flask_wtf.csrf import CSRFProtect
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import boto3

app = Flask(__name__)
csrf = CSRFProtect(app)

# We are intialized AWS variable
AWS_REGION = 'Global'
USER_POOL_ID = ''
APP_CLIENT_ID = ''
IDENTITY_POOL_ID = ''

# We are Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# We are initialize Congito client
cognito_client = boto3.client('cognito-idp', region_name=AWS_REGION)

# We are define the route
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        email = request.form["email"]
        
        # Step 1: We are create  a new user in Cognito User Pool
        try:
            response = cognito_client.sign_up(
                ClientId=APP_CLIENT_ID
                Username = username,
                Password=password,
                UserAttributes=[
                    {'Name': 'email', 'Value': email},
                ],
            )
            user_sub = response['UserSub']
            session['user_sub'] = user_sub
            return redirect(url_for('mfa_verification'))
        except cognito_client.exceptions.UsernameExistsException:
            return render_template("signup.html", error="Username already exists. Choose a different one")
        except cognito_client.exceptions.UserLambdaValidationException as e:
            return render_template("signup.html", error=f"Error: {e}")
        except Exception as e:
            return render_template("signup.html", error=f"Error: {e}")
        
    return render_template("signup.html")

@app.route("/mfa_verification", methods=["GET", "POST"])
def mfa_verification():
    if 'user_sub' not in session:
        return redirect(url_for('signup'))
    if request.method =="POST":
        mfa_code = request.form["mfa_code"]
        
        # Step 2: Verify MFA code
        try:
            cognito_client.verify_software_token(
                AccessToken=session['user_sub'],
                UserCode = mfa_code,
            )

            return render_template("index.html", prompt="", generated_text="MFA verification successful.")
        except cognito_client.exceptions.CodeMismatchException:
            return render_template("mfa_verification.html", error="Incorrect MFA code. Please try again.")
        
        except Exception as e:
            return render_template("mfa_verification.html", error=f"Error:{e}")
        
    return render_template("mfa_verification.html")

if __name__ == "__main__":
    app.secret_key = 'supersecretkey'
    app.run(debug=True)
        
