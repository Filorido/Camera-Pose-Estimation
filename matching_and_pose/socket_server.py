# socket_server.py
import socket, struct, json

class JSONSocketOneShot:
    """
    Apre un socket TCP, accetta un client, invia un singolo dict JSON length-prefixed, poi chiude.
    """
    def __init__(self, host='127.0.0.1', port=5005):
        self.host = host
        self.port = port

    def send_once(self, data: dict):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(1)
        print(f"[SocketOneShot] In ascolto su {self.host}:{self.port}")
        conn, addr = server.accept()
        print(f"[SocketOneShot] Connessione da {addr}, invio dati…")
        payload = json.dumps(data).encode('utf-8')
        conn.sendall(struct.pack('>I', len(payload)))
        conn.sendall(payload)
        conn.close()
        server.close()
        print("[SocketOneShot] Dati inviati e socket chiuso")
