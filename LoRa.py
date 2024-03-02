# Import the require library
from network import LoRa
import socket
import time

# We are initialize LoRa module
lora = LoRa(model=LoRa.LORA, frequency=915000000)

# We are Create LoRa socket
s = socket.socket(socket.AF_LORA, socket.SOCK_RAW)

# We are set the LoR socket blocking to False
s.setblocking(False)

# We are starting an while loop
while True:
    # We are sending some data
    s.send("Hello LoRa")
    
    # We are wait for incoming messages
    data = s.recv(64)
    if data:
        print("Received", data)
        
    time.sleep(2)
