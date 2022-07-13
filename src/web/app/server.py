from flask import Flask #importing the module
import folium
app=Flask(__name__) #instantiating flask object

@app.route('/') #defining a route in the application
def hello_world():
    return render_template("index.html")

if __name__=='__main__': #calling  main
    print(__name__)
    app.debug=True #setting the debugging option for the application instance
    app.run() #launching the flask's integrated development webserver

