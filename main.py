from swiftai import SwiftAI

if __name__ == "__main__":
    swift = SwiftAI('saved_vars/trained_swiftai_songs_model.pth')
    for song in swift.write_song("Shake it off man"):
        print("\n" + song + "\n")

