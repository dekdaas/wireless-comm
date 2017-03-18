Wireless File Transfer with audio. By Linh Albert Pham, Michelle Ahn, Ray Ramamurti

Emits chirps to be recorded by a listening computer and prints the received message.

USAGE: python -i music.py
receive(duration)
	Records audio for duration seconds, processes, then prints the message and writes the message to a file. Default file is 'received.txt'.

transmit(filename)
	Reads a standard ASCII text file, then plays the encoded audio.
