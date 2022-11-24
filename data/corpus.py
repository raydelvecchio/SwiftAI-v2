import os
import re
import json


def get_json_filenames(directory=None):
    """
    Returns a list of .json filenames of Taylor Swift lyrics downloaded from Genius. Filters them to ensure only
    .json files are returned for preprocessing. Does this with os.listdir (list directory) and os.getcwd (get current
    working directory). Extensible even if we download more lyrics too!
    """
    filetype = re.compile('^.*\.(json)$')
    if directory is not None:
        all_files = os.listdir(directory)
    else:
        all_files = all_files = os.listdir(os.getcwd())
    return [file for file in all_files if filetype.match(file)]


def clean_song(song: str) -> str:
    """
    Cleans the input song (string). Removes first/last line of the input song from the .json files. Removes first line
    because it's always formatted as "songname Lyrics", and last line because it has embedded ads for some reason.
    Filters out double newlines and other special chars we don't need.
    """
    song = song.replace('\n\n', '\n')
    for c in ['\"', '(', ')']:
        song = song.replace(c, "")
    song = song.replace("See Taylor Swift LiveGet tickets as low as $307You might also like", "")  # metatext bad!
    return song[song.find('\n') + 1: song.rfind('\n')]


def generate_corpus():
    """
    For all .json files in the /data folder, add all song lyrics into a single .txt file to use as our corpus. UTF-8
    encoding solves character map encoding errors, since there are some non-text characters in our .json files. Should
    only be called once, unless we re-download with new data. Meant to be run locally from this file!
    """
    album_files = get_json_filenames()
    with open('lyrics.txt', 'w', encoding="utf-8") as f:
        for album in album_files:
            with open(album) as lyr:
                j = json.load(lyr)
            tracks = j['tracks']
            for track in tracks:
                lyrics = track['song']['lyrics']
                lyrics = clean_song(lyrics)
                f.write(lyrics + '\n')


if __name__ == "__main__":
    generate_corpus()
