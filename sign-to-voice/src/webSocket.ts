// useWebSocket.ts
import { useEffect, useRef } from 'react';

export function useWebSocket(onMessage: (data: any) => void) {
  const socket = useRef<WebSocket | null>(null);

  useEffect(() => {
    socket.current = new WebSocket("ws://localhost:8000/ws");
    socket.current.onmessage = (event) => onMessage(JSON.parse(event.data));

    return () => socket.current?.close();
  }, []);

  const sendFrame = (data: string | Blob) => {
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(data);
    }
  };

  return { sendFrame };
}
