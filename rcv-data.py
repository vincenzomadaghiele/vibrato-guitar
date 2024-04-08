from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def liveFeaturesIn_handler(address, *args):
	print(f"{address}: {args}")

	# check which feature is received
	feature = address.split('/')[-1]
	if feature == 'loudness':
		loudness = np.array(args).tolist()
		print(f"state: {loudness}")
		result = process(loudness)
		sendControlsToPD(result, client)


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")

if __name__ == '__main__': 

	# DEFINE
	ip = "192.168.210.36" # localhost
	port_rcv = 9999 # receive port from PD

	dispatcher = Dispatcher()
	dispatcher.map("/*", liveFeaturesIn_handler)
	dispatcher.set_default_handler(default_handler)

	# define server
	server = BlockingOSCUDPServer((ip, port_rcv), dispatcher)
	server.serve_forever()  # Blocks forever



