from swiftai import SwiftAI
import argparse

parser = argparse.ArgumentParser(prog='SwiftAI!',
                                 description='This program interfaces with SwiftAI to generate Taylor Swift songs based on defined parameters.',
                                 epilog='Have fun generating!')

parser.add_argument('-p', '--prompt', help='Prompt to kickstart generation of songs')
parser.add_argument('-l', '--length', help='Length of songs')
parser.add_argument('-r', '--return', help='Number of songs to return from SwiftAI')
parser.add_argument('-t', '--temperature', help="Default starting temperature of song; each new song decreases temp")
parser.add_argument('-c', '--cpu', help="If we should use CPU or not; 1 means use CPU, 0 means use GPU.")

args = parser.parse_args()

# if no arguments are given, resort back to the default values for these!
prompt = getattr(args, 'prompt') if getattr(args, 'prompt') is not None else "<|endoftext|>"
length = getattr(args, 'length') if getattr(args, 'length') is not None else 300
num_ret = getattr(args, 'return') if getattr(args, 'return') is not None else 3
temp = getattr(args, 'temperature') if getattr(args, 'temperature') is not None else 1.5
cpu = getattr(args, 'cpu') if getattr(args, 'cpu') is not None else 1  # default is using CPU

use = False
if cpu == 1:
    use = False
elif cpu == 0:
    use = True


swift = SwiftAI('saved_vars/trained_swiftai_songs_model.pth', use_gpu=use)
for song in swift.write_song(prompt, length=length, num_ret=num_ret, max_temp=temp):
    print("\n" + song + "\n")
