from flask import Flask, jsonify
from swiftai import SwiftAI

app = Flask("SwiftAI")
swift = SwiftAI('saved_vars/trained_swiftai_songs_model.pth')


@app.route('/makesong/')
def makesong():
    song = swift.write_song("hello", num_ret=1)[0]
    return jsonify({'song': song})


app.run()
