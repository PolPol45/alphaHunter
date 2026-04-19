import zmq

def main():
    context = zmq.Context(1)
    
    # Frontend riceve i messaggi dai Publisher (es. DataProducer)
    frontend = context.socket(zmq.XSUB)
    frontend.bind("tcp://*:5555")

    # Backend inoltra i messaggi a tutti i Subscriber in ascolto
    backend = context.socket(zmq.XPUB)
    backend.bind("tcp://*:5556")
    
    print("🚀 [ZMQ Event Broker] avviato e in ascolto. (Frontend: 5555 | Backend: 5556)")
    try:
        zmq.proxy(frontend, backend)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[{e}] Errore Broker ZMQ")
    finally:
        frontend.close()
        backend.close()
        context.term()

if __name__ == "__main__":
    main()
