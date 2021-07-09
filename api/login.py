@app.route('/authenticate', methods=['GET', 'POST'])
@cross_origin()
def authenticate():
 
     if request.method == "POST":
         print("post")
         print(request.json)
         # request.form doesn't work for json data ???
         username = request.json["username"]
         password = request.json["password"]
         rows = util.fetch("select * from credentials where username like '{}' and password like '{}'".format(username, password))
         if len(rows) != 0:
            return jsonify({}), 200
 
         return jsonify({}), 404
 
     return jsonify({}), 400
 

