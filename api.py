from flask import Flask, jsonify, request
from swiftai import SwiftAI

app = Flask("SwiftAI")
swift = SwiftAI('saved_vars/trained_swiftai_songs_model.pth', use_gpu=False)  # server may not have GPU; takes longer


@app.route('/songwrite', methods=['GET'])
def generate_song():
    """
    When the /songwrite endpoint is accessed, we parse the arguments given in the API call and send back all the
    songs written by SwiftAI in a JSON format. Can access with the following URL if ran locally:
    http://127.0.0.1:5000/songwrite?prompt=PROMPT&length=LENGTH&num=NUM&temp=TEMP
    """
    args = request.args

    prompt = args.get('prompt') if args.get('prompt') is not None else "<|endoftext|>"
    length = int(args.get('length')) if args.get('length') is not None else 300
    num = int(args.get('num')) if args.get('num') is not None else 3
    temp = float(args.get('temp')) if args.get('temp') is not None else 1.5

    written_songs = swift.write_song(prompt, length=length, num_ret=num, max_temp=temp)

    return jsonify(written_songs), 200


app.run()
