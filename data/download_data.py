import lyricsgenius
from constants import GENIUS_ACCESS_TOKEN, ALBUMS


def get_genius():
    """
    Gets the Genius Lyrics opbject complete with hyperparameters to download lyrics for. Currently configured to
    remove headers (things like [Chorus], [Verse], etc) and only songs (not skits / stories) from the songs.
    :return:
    """
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
    genius.remove_section_headers = True
    genius.skip_non_songs = True
    return genius


def download_lyrics_from_albums(albums=ALBUMS, artist='Taylor Swift'):
    """
    Downloads all lyrics from all albums into JSON files from GENIUS API. Should be called only once.
    """
    genius = get_genius()
    for album in albums:
        downloaded = genius.search_album(album, artist)
        downloaded.save_lyrics()


def download_lyrics_from_artist(artist="Taylor Swift"):
    """
    Downloads all lyrics from all songs an artist makes into one JSON file. Should also only be called once.
    This will take a lot longer than specifying specific albums. Prone to timing out if connection is bad.
    """
    genius = get_genius()
    artist_data = genius.search_artist(artist)
    artist_data.save_lyrics()


if __name__ == "__main__":
    download_lyrics_from_albums()
